import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
training/train.py

Phase 1: python training/train.py            (frozen encoders, uses cache)
Phase 2: python training/train.py --phase2   (unfrozen top layers, raw data)

Changes vs previous version:
  - Phase 2 optimiser uses differential LR: encoder layers get lr * 0.1,
    head/fusion/router layers keep the full phase2 LR. Prevents catastrophic
    forgetting while still allowing encoder adaptation.
  - Removed hardcoded phase2_max_steps=100 and phase2_lr_divisor=20 defaults
    that were silently overriding train.yaml values (yaml now always wins).
  - unfreeze_top_layers handles n_layers=0 (projection-only unfreeze).
  - compute_class_weights smooth=True uses sqrt for Phase 1 cached path.
"""

import argparse, random, time
from collections import Counter
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

def load_config(config_dir):
    merged = {}
    for name in ("base", "model", "train"):
        with open(Path(config_dir) / f"{name}.yaml") as f:
            merged.update(yaml.safe_load(f))
    return merged

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def cache_exists():
    return all((FEATURE_DIR / f"{ds}_{sp}.pt").exists()
               for ds in ("wrime", "ravdess")
               for sp in ("train", "validation", "test"))


# ── Class weights ──────────────────────────────────────────────────────────────

def compute_class_weights(loaders, device, smooth: bool = False):
    """
    smooth=True (Phase 1): sqrt-smooth weights because WeightedRandomSampler
    already balances class exposure — full inverse-freq on top would
    double-correct and inflate minority-class recall at the cost of precision.

    smooth=False (Phase 2): full inverse-frequency, no sampler active.
    """
    counts = Counter()
    for mod in ("wrime", "ravdess"):
        for batch in loaders["train"][mod]:
            counts.update(batch["label"].tolist())
    total = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        weights[i] = total / (NUM_CLASSES * max(counts.get(i, 1), 1))
    weights = weights / weights.mean()

    if smooth:
        weights = torch.sqrt(weights)
        weights = weights / weights.mean()

    print(f"Class weights {'(sqrt-smoothed)' if smooth else '(full inverse-freq)'}:")
    for i, (cls, w) in enumerate(zip(EMOTION_CLASSES, weights)):
        print(f"  {cls:12s} {w:.3f}  (n={counts.get(i,0):,})")
    return weights.to(device)


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

def process_batch(batch, model, device, config, metrics, class_weights=None):
    clf_w = config["training"]["loss"]["classification_weight"]
    reg_w = config["training"]["loss"]["regression_weight"]
    def d(x): return x.to(device, non_blocking=True) if x is not None else None

    labels  = d(batch["label"])
    valence = d(batch["valence"])
    arousal = d(batch["arousal"])

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
        class_weights=class_weights,
    )
    pred = outputs["predicted_class"]
    metrics["f1_macro"].update(pred, labels)
    metrics["f1_weighted"].update(pred, labels)
    metrics["accuracy"].update(pred, labels)
    metrics["rmse_val"].update(outputs["valence"].float(), valence.float())
    metrics["rmse_aro"].update(outputs["arousal"].float(), arousal.float())
    return loss_dict["total"]


# ── Train epoch ────────────────────────────────────────────────────────────────

def train_epoch(model, loaders, optimizer, scheduler,
                device, config, writer, epoch, gs, class_weights, phase2=False):
    model.train()
    metrics = build_metrics(device)
    clip    = config["training"]["gradient_clip"]
    log_n   = config["training"]["log_every_n_steps"]
    accum   = config["training"].get("grad_accumulation_steps", 1)
    tc      = config["training"]

    if phase2:
        max_steps = tc.get("phase2_max_steps", 75)
        rav_ratio = tc.get("phase2_ravdess_ratio", 0)
    else:
        max_steps = tc.get("max_steps_per_epoch", None)
        rav_ratio = tc.get("ravdess_ratio", 1)

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
        if rav_ratio == 0:
            batch = next(wrime_it)
        elif (step % (rav_ratio + 1)) < rav_ratio:
            batch = next(ravdess_it)
        else:
            batch = next(wrime_it)

        loss = process_batch(batch, model, device, config, metrics, class_weights)
        (loss / accum).backward()

        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         lr=f"{scheduler.get_last_lr()[0]:.1e}")
        if gs[0] % log_n == 0:
            writer.add_scalar("train/loss_step", loss.item(), gs[0])
        gs[0] += 1

    res = compute_metrics(metrics)
    res["loss"] = epoch_loss / n_steps
    writer.add_scalar("train/loss_epoch",  res["loss"],     epoch)
    writer.add_scalar("train/f1_macro",    res["f1_macro"], epoch)
    return res


# ── Eval epoch ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loaders, split, device, config, writer, epoch, class_weights):
    model.eval()
    metrics = build_metrics(device)
    total_loss, total_batches = 0.0, 0
    for mod in ("wrime", "ravdess"):
        for batch in tqdm(loaders[split][mod],
                          desc=f"Ep {epoch:02d} [{split}/{mod}]",
                          leave=False, dynamic_ncols=True):
            loss = process_batch(batch, model, device, config, metrics, class_weights)
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
    print(f"  Saved → {path.name}")


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


# ── Unfreeze top encoder layers (phase 2) ─────────────────────────────────────

def unfreeze_top_layers(model, n_layers=1):
    for encoder, name in [(model.text_encoder, "BERT"),
                          (model.audio_encoder, "Wav2Vec2")]:
        if n_layers > 0:
            try:
                layers = encoder.encoder.encoder.layer
            except AttributeError:
                layers = encoder.encoder.encoder.layers
            for layer in layers[-n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # Always unfreeze the projection regardless of n_layers
        for p in encoder.projection.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  {name}: {n_layers} transformer layer(s) + projection unfrozen ({n:,} params)")


# ── Build phase 2 optimiser with differential LR ──────────────────────────────

def build_phase2_optimizer(model, lr, weight_decay):
    """
    Encoder layers (just unfrozen) get lr * 0.1 — small enough to adapt
    without overwriting Phase 1 representations (catastrophic forgetting).
    All other trainable params (head, fusion, router) keep the full lr.
    """
    encoder_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_encoder" in name or "audio_encoder" in name:
            encoder_params.append(p)
        else:
            other_params.append(p)

    n_enc   = sum(p.numel() for p in encoder_params)
    n_other = sum(p.numel() for p in other_params)
    print(f"  Encoder params: {n_enc:,} @ lr={lr*0.1:.1e}")
    print(f"  Other params:   {n_other:,} @ lr={lr:.1e}")

    return AdamW(
        [
            {"params": encoder_params, "lr": lr * 0.1},
            {"params": other_params,   "lr": lr},
        ],
        weight_decay=weight_decay,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/")
    parser.add_argument("--device",   default=None)
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--phase2",   action="store_true")
    parser.add_argument("--no-cache", action="store_true")
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
    writer = SummaryWriter(log_dir=str(log_dir) + ("_p2" if args.phase2 else "_p1"))

    # ── DataLoaders ────────────────────────────────────────────────────────────
    use_cache = cache_exists() and not args.no_cache and not args.phase2
    if use_cache:
        print("Phase 1 — cached features (fast)")
        loaders = build_cached_dataloaders(
            batch_size=config["training"].get("cached_batch_size", 512)
        )
    else:
        print(f"Phase {'2' if args.phase2 else '1'} — raw dataloaders")
        loaders = build_dataloaders(
            config, num_workers=0, pin_memory=(device.type == "cuda")
        )

    # ── Model ──────────────────────────────────────────────────────────────────
    print("Initialising model...")
    model = MoEEmotionModel(config).to(device)

    if args.phase2:
        p1_ckpt = ckpt_dir / "best.pt"
        if p1_ckpt.exists():
            print(f"Loading phase 1 weights from {p1_ckpt}")
            model.load_state_dict(
                torch.load(p1_ckpt, map_location=device)["model"]
            )
        n_unfreeze = config["training"].get("phase2_unfreeze_layers", 1)
        print(f"Unfreezing top {n_unfreeze} encoder layer(s)...")
        unfreeze_top_layers(model, n_layers=n_unfreeze)
    elif use_cache:
        for p in model.text_encoder.parameters():  p.requires_grad = False
        for p in model.audio_encoder.parameters(): p.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")

    # ── Class weights ──────────────────────────────────────────────────────────
    class_weights = compute_class_weights(loaders, device, smooth=use_cache)

    # ── Optimiser ──────────────────────────────────────────────────────────────
    tc     = config["training"]
    accum  = tc.get("grad_accumulation_steps", 1)
    epochs = tc.get("phase2_epochs", 15) if args.phase2 else tc["epochs"]

    # LR: always read divisor from yaml — no hardcoded override
    lr_div = tc.get("phase2_lr_divisor", 40)
    lr     = tc["learning_rate"] / lr_div if args.phase2 else tc["learning_rate"]

    max_s    = tc.get("phase2_max_steps", 75) if args.phase2 else tc.get("max_steps_per_epoch", None)
    n_base   = max(len(loaders["train"]["wrime"]), len(loaders["train"]["ravdess"]))
    steps_ep = min(n_base, max_s) if max_s else n_base
    total_s  = (steps_ep // accum) * epochs

    if args.phase2:
        print("Building phase 2 optimiser (differential LR)...")
        optimizer = build_phase2_optimizer(model, lr, tc["weight_decay"])
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=tc["weight_decay"],
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(tc["warmup_steps"], total_s // 10),
        num_training_steps=max(total_s, 1),
    )

    start_epoch, gs = 1, [0]
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1

    es_cfg    = tc["early_stopping"]
    stopper   = EarlyStopping(es_cfg["patience"], es_cfg["mode"])
    primary   = es_cfg["metric"].replace("val_", "")
    best_name = "best_p2.pt" if args.phase2 else "best.pt"

    print(f"\n{'='*50}")
    print(f"Phase        : {'2 (fine-tune)' if args.phase2 else '1 (frozen)'}")
    print(f"Epochs       : {epochs}  |  Steps/epoch: {steps_ep}")
    print(f"LR           : {lr:.1e} (divisor {lr_div if args.phase2 else 1})")
    print(f"Batch        : {tc.get('cached_batch_size', 512) if use_cache else tc['batch_size']}")
    print(f"Weights      : {'sqrt-smoothed (sampler active)' if use_cache else 'full inverse-freq'}")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, loaders, optimizer, scheduler,
                         device, config, writer, epoch, gs,
                         class_weights, phase2=args.phase2)
        vr = eval_epoch(model, loaders, "val", device, config,
                        writer, epoch, class_weights)
        print(
            f"Ep {epoch:02d}/{epochs} | {time.time()-t0:.0f}s | "
            f"train loss {tr['loss']:.3f} f1 {tr['f1_macro']:.3f} | "
            f"val loss {vr['loss']:.3f} f1 {vr['f1_macro']:.3f} "
            f"acc {vr['accuracy']:.3f}"
        )
        save_ckpt(model, optimizer, scheduler, epoch, vr, ckpt_dir / "last.pt")
        if stopper.step(vr[primary]):
            save_ckpt(model, optimizer, scheduler, epoch, vr, ckpt_dir / best_name)
            print(f"  ✓ Best {primary}: {vr[primary]:.4f}")
        if stopper.triggered:
            print(f"Early stopping at epoch {epoch}"); break

    # ── Test eval ──────────────────────────────────────────────────────────────
    print(f"\nTest evaluation ({best_name})...")
    model.load_state_dict(
        torch.load(ckpt_dir / best_name, map_location=device)["model"]
    )
    test_res = eval_epoch(model, loaders, "test", device, config,
                          writer, 0, class_weights)
    print("\nTest results:")
    for k, v in test_res.items(): print(f"  {k:20s}: {v:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mod in ("wrime", "ravdess"):
            for batch in loaders["test"][mod]:
                def d(x): return x.to(device) if x is not None else None
                if "text_features" in batch:
                    kwargs = {"text_features":  d(batch["text_features"]),
                              "audio_features": d(batch["audio_features"])}
                else:
                    kwargs = {k: d(v) for k, v in batch_to_model_inputs(batch).items()}
                out = model(**kwargs)
                all_preds.extend(out["predicted_class"].cpu().tolist())
                all_labels.extend(batch["label"].tolist())

    print("\nPer-class F1:")
    print(classification_report(all_labels, all_preds,
                                 target_names=EMOTION_CLASSES, digits=3))
    writer.close()
    print(f"Checkpoints: {ckpt_dir}")

if __name__ == "__main__":
    main()