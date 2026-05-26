import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
training/train.py

Optimised training loop — uses cached encoder features when available.

Step 1 (run once):  python training/cache_features.py   (~10 min)
Step 2 (training):  python training/train.py            (~1-2 min/epoch)

When cache exists, encoders are bypassed entirely each step.
Only router + fusion + heads (~3M params) are trained.
"""

import argparse, random, time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report

from data.cached_dataset   import build_cached_dataloaders, FEATURE_DIR
from data.combined_dataset import build_dataloaders, batch_to_model_inputs
from data.emotion_mappings import NUM_CLASSES, EMOTION_CLASSES
from models.multimodal.moe import MoEEmotionModel


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_dir: str) -> dict:
    merged = {}
    for name in ("base", "model", "train"):
        with open(Path(config_dir) / f"{name}.yaml") as f:
            merged.update(yaml.safe_load(f))
    return merged


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cache_exists() -> bool:
    required = [
        FEATURE_DIR / "wrime_train.pt",
        FEATURE_DIR / "ravdess_train.pt",
        FEATURE_DIR / "wrime_validation.pt",
        FEATURE_DIR / "ravdess_validation.pt",
        FEATURE_DIR / "wrime_test.pt",
        FEATURE_DIR / "ravdess_test.pt",
    ]
    return all(p.exists() for p in required)


# ── Metrics ────────────────────────────────────────────────────────────────────

def build_metrics(device):
    return {
        "f1_macro":    MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device),
        "f1_weighted": MulticlassF1Score(num_classes=NUM_CLASSES, average="weighted").to(device),
        "accuracy":    MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device),
        "rmse_val":    MeanSquaredError(squared=False).to(device),
        "rmse_aro":    MeanSquaredError(squared=False).to(device),
    }


def compute_metrics(m):
    return {
        "f1_macro":     m["f1_macro"].compute().item(),
        "f1_weighted":  m["f1_weighted"].compute().item(),
        "accuracy":     m["accuracy"].compute().item(),
        "rmse_valence": m["rmse_val"].compute().item(),
        "rmse_arousal": m["rmse_aro"].compute().item(),
    }


# ── Batch forward ──────────────────────────────────────────────────────────────

def process_batch(batch, model, device, config, metrics):
    clf_w = config["training"]["loss"]["classification_weight"]
    reg_w = config["training"]["loss"]["regression_weight"]

    def d(x): return x.to(device, non_blocking=True) if x is not None else None

    labels  = d(batch["label"])
    valence = d(batch["valence"])
    arousal = d(batch["arousal"])

    # Cached path: features already computed
    # Raw path:    input_ids / audio_values present instead
    kwargs = {}
    if "text_features" in batch:
        kwargs["text_features"]  = d(batch["text_features"])
        kwargs["audio_features"] = d(batch["audio_features"])
    else:
        kwargs["input_ids"]      = d(batch.get("input_ids"))
        kwargs["attention_mask"] = d(batch.get("attention_mask"))
        kwargs["token_type_ids"] = d(batch.get("token_type_ids"))
        kwargs["audio_values"]   = d(batch.get("audio_values"))

    outputs   = model(**kwargs)
    loss_dict = model.head.compute_loss(
        outputs, labels, valence, arousal,
        clf_weight=clf_w, reg_weight=reg_w,
    )

    pred = outputs["predicted_class"]
    metrics["f1_macro"].update(pred, labels)
    metrics["f1_weighted"].update(pred, labels)
    metrics["accuracy"].update(pred, labels)
    metrics["rmse_val"].update(outputs["valence"].float(), valence.float())
    metrics["rmse_aro"].update(outputs["arousal"].float(), arousal.float())

    return loss_dict["total"]


# ── Train epoch ────────────────────────────────────────────────────────────────

def train_epoch(model, loaders, optimizer, scheduler, device, config, writer, epoch, gs):
    model.train()
    metrics   = build_metrics(device)
    clip      = config["training"]["gradient_clip"]
    log_n     = config["training"]["log_every_n_steps"]
    accum     = config["training"].get("grad_accumulation_steps", 1)
    max_steps = config["training"].get("max_steps_per_epoch", None)

    wrime_it   = cycle(loaders["train"]["wrime"])
    ravdess_it = cycle(loaders["train"]["ravdess"])
    n_base     = max(len(loaders["train"]["wrime"]),
                     len(loaders["train"]["ravdess"]))
    n_steps    = min(n_base, max_steps) if max_steps else n_base

    epoch_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(range(n_steps), desc=f"Ep {epoch:02d} [train]",
                leave=False, dynamic_ncols=True)

    for step in pbar:
        batch = next(wrime_it) if step % 2 == 0 else next(ravdess_it)
        loss  = process_batch(batch, model, device, config, metrics)
        (loss / accum).backward()

        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         lr=f"{scheduler.get_last_lr()[0]:.1e}")
        if gs[0] % log_n == 0:
            writer.add_scalar("train/loss_step", loss.item(), gs[0])
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], gs[0])
        gs[0] += 1

    res = compute_metrics(metrics)
    res["loss"] = epoch_loss / n_steps
    writer.add_scalar("train/loss_epoch",  res["loss"],     epoch)
    writer.add_scalar("train/f1_macro",    res["f1_macro"], epoch)
    return res


# ── Eval epoch ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loaders, split, device, config, writer, epoch):
    model.eval()
    metrics = build_metrics(device)
    total_loss, total_batches = 0.0, 0

    for mod in ("wrime", "ravdess"):
        for batch in tqdm(loaders[split][mod],
                          desc=f"Ep {epoch:02d} [{split}/{mod}]",
                          leave=False, dynamic_ncols=True):
            loss = process_batch(batch, model, device, config, metrics)
            total_loss += loss.item(); total_batches += 1

    res = compute_metrics(metrics)
    res["loss"] = total_loss / max(total_batches, 1)
    for k, v in res.items():
        writer.add_scalar(f"{split}/{k}", v, epoch)
    return res


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_ckpt(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics}, path)


class EarlyStopping:
    def __init__(self, patience, mode="max"):
        self.patience = patience; self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0; self.triggered = False

    def step(self, v):
        improved = (self.mode == "max" and v > self.best) or \
                   (self.mode == "min" and v < self.best)
        if improved: self.best = v; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.triggered = True
        return improved


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="Force raw dataloaders even if cache exists")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    log_dir  = Path(config["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # ── DataLoaders ────────────────────────────────────────────────────────────
    use_cache = cache_exists() and not args.no_cache
    if use_cache:
        print("Cache found — using pre-computed features (fast mode)")
        batch_size = config["training"].get("cached_batch_size", 512)
        loaders = build_cached_dataloaders(batch_size=batch_size)
    else:
        print("No cache — using raw dataloaders (run cache_features.py for speed)")
        print("Hint: python training/cache_features.py")
        loaders = build_dataloaders(
            config, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # ── Model ──────────────────────────────────────────────────────────────────
    print("Initialising MoEEmotionModel...")
    model = MoEEmotionModel(config).to(device)

    # In cached mode, freeze both encoders — training only touches
    # router + fusion + heads
    if use_cache:
        for p in model.text_encoder.parameters():  p.requires_grad = False
        for p in model.audio_encoder.parameters(): p.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")

    # ── Optimiser ──────────────────────────────────────────────────────────────
    tc      = config["training"]
    accum   = tc.get("grad_accumulation_steps", 1)
    max_s   = tc.get("max_steps_per_epoch", None)
    n_base  = max(len(loaders["train"]["wrime"]),
                  len(loaders["train"]["ravdess"]))
    steps_ep = min(n_base, max_s) if max_s else n_base
    total_s  = (steps_ep // accum) * tc["epochs"]

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tc["warmup_steps"],
        num_training_steps=total_s,
    )

    start_epoch, gs = 1, [0]
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    es_cfg  = tc["early_stopping"]
    stopper = EarlyStopping(es_cfg["patience"], es_cfg["mode"])
    primary = es_cfg["metric"].replace("val_", "")

    mode_str = "CACHED (fast)" if use_cache else "RAW (slow — run cache_features.py)"
    print(f"\nMode            : {mode_str}")
    print(f"Steps/epoch     : {steps_ep}  |  Epochs: {tc['epochs']}")
    print(f"Batch size      : {tc['batch_size'] if not use_cache else tc.get('cached_batch_size', 512)}"
          f"  (effective: ×{accum})")
    print(f"Trainable params: {n_train:,}\n")

    for epoch in range(start_epoch, tc["epochs"] + 1):
        t0 = time.time()
        tr = train_epoch(model, loaders, optimizer, scheduler,
                         device, config, writer, epoch, gs)
        vr = eval_epoch(model, loaders, "val", device, config, writer, epoch)
        elapsed = time.time() - t0

        print(
            f"Ep {epoch:02d}/{tc['epochs']} | {elapsed:.0f}s | "
            f"train loss {tr['loss']:.3f} f1 {tr['f1_macro']:.3f} | "
            f"val loss {vr['loss']:.3f} f1 {vr['f1_macro']:.3f} "
            f"acc {vr['accuracy']:.3f}"
        )
        save_ckpt(model, optimizer, scheduler, epoch, vr, ckpt_dir / "last.pt")

        if stopper.step(vr[primary]):
            save_ckpt(model, optimizer, scheduler, epoch, vr, ckpt_dir / "best.pt")
            print(f"  ✓ Best {primary}: {vr[primary]:.4f}")

        if stopper.triggered:
            print(f"Early stopping at epoch {epoch}"); break

    # ── Test eval ──────────────────────────────────────────────────────────────
    print("\nTest evaluation...")
    model.load_state_dict(
        torch.load(ckpt_dir / "best.pt", map_location=device)["model"]
    )
    tr_res = eval_epoch(model, loaders, "test", device, config, writer, epoch=0)
    print("\nTest results:")
    for k, v in tr_res.items(): print(f"  {k:20s}: {v:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mod in ("wrime", "ravdess"):
            for batch in loaders["test"][mod]:
                kwargs = {}
                if "text_features" in batch:
                    kwargs["text_features"]  = batch["text_features"].to(device) if batch["text_features"] is not None else None
                    kwargs["audio_features"] = batch["audio_features"].to(device) if batch["audio_features"] is not None else None
                else:
                    kwargs = {k: v.to(device) if v is not None else None
                              for k, v in batch_to_model_inputs(batch).items()}
                out = model(**kwargs)
                all_preds.extend(out["predicted_class"].cpu().tolist())
                all_labels.extend(batch["label"].tolist())

    print("\nPer-class F1:")
    print(classification_report(all_labels, all_preds,
                                 target_names=EMOTION_CLASSES, digits=3))
    writer.close()
    print(f"Checkpoints: {ckpt_dir} | TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()