# train.py (FINAL CLEAN VERSION)

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deap_dataset import DEAPDataset, loso_split
from deap_emotion_model import DEAPEmotionModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHYSIO_CSV = str(SRC_DIR / "data" / "physio_features.csv")
AUDIO_CSV  = str(SRC_DIR / "data" / "audio_features.csv")
MAT_DIR    = str(SRC_DIR.parent / "db" / "deap-dataset" / "data_preprocessed_python")

CKPT_DIR = str(SRC_DIR / "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _aggregate_val_outputs(val_outputs):
    agg = {"mean_rmse": [], "mean_ccc": []}

    dims = list(val_outputs[0].rmse.keys())

    for dim in dims:
        agg[f"rmse_{dim}"] = []
        agg[f"ccc_{dim}"]  = []

    for out in val_outputs:
        agg["mean_rmse"].append(out.mean_rmse)
        agg["mean_ccc"].append(out.mean_ccc)

        for dim in dims:
            agg[f"rmse_{dim}"].append(out.rmse[dim])
            agg[f"ccc_{dim}"].append(out.ccc[dim])

    return {k: float(np.mean(v)) for k, v in agg.items()}


def _format_metrics(metrics):
    parts = [
        f"CCC={metrics['mean_ccc']:.4f}",
        f"RMSE={metrics['mean_rmse']:.4f}",
    ]

    for k, v in metrics.items():
        if k.startswith("ccc_") and k != "mean_ccc":
            dim = k.replace("ccc_", "")
            parts.append(f"{dim}_ccc={v:.3f}")

    return " | ".join(parts)


def save_checkpoint(model, path, config, metrics, epoch, best_val_ccc):
    obj = {
        "state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
        "training_info": {
            "epoch": epoch,
            "best_val_ccc": best_val_ccc,
        },
    }
    torch.save(obj, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_fold(
    dataset,
    test_subject,
    val_subject,
    *,
    embed_dim,
    dropout,
    lb_lambda,
    lr,
    weight_decay,
    batch_size,
    max_epochs,
    patience,
    device,
    seq_model,
):

    print(f"\n{'='*60}")
    print(f"FOLD {test_subject} | val={val_subject} | model={seq_model.upper()}")
    print(f"{'='*60}")

    splits = loso_split(dataset, test_subject, val_subject)
    dataset.fit_normalizer(splits["train"].indices)

    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(splits["val"], batch_size=batch_size, shuffle=False)

    model = DEAPEmotionModel(
        embed_dim=embed_dim,
        dropout=dropout,
        lb_lambda=lb_lambda,
        sequence_model_type=seq_model,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_ccc = -1e9
    patience_ctr = 0

    # checkpoint paths
    fold_path = os.path.join(CKPT_DIR, f"{seq_model}_fold{test_subject}.pt")
    best_path = os.path.join(CKPT_DIR, f"{seq_model}_best.pt")

    for epoch in range(1, max_epochs + 1):

        # ================= TRAIN =================
        model.train()
        train_losses = []

        for physio, audio, mask, targets in train_loader:
            physio, audio, mask = physio.to(device), audio.to(device), mask.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda")):
                out = model.training_step(physio, audio, mask, targets)

            scaler.scale(out.loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(out.loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # ================= VALIDATION =================
        model.eval()
        val_outputs = []

        with torch.no_grad():
            for physio, audio, mask, targets in val_loader:
                physio, audio, mask = physio.to(device), audio.to(device), mask.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}

                with autocast(enabled=(device.type == "cuda")):
                    val_out = model.validation_step(physio, audio, mask, targets)

                val_outputs.append(val_out)

        val_metrics = _aggregate_val_outputs(val_outputs)
        val_ccc = val_metrics["mean_ccc"]

        scheduler.step()

        # ================= SAVE =================
        config = {
            "seq_model": seq_model,
            "embed_dim": embed_dim,
            "dropout": dropout,
            "lb_lambda": lb_lambda,
        }

        if val_ccc > best_val_ccc + 1e-3:
            best_val_ccc = val_ccc
            patience_ctr = 0

            save_checkpoint(model, fold_path, config, val_metrics, epoch, best_val_ccc)
            save_checkpoint(model, best_path, config, val_metrics, epoch, best_val_ccc)

            marker = "✓ BEST"

        else:
            patience_ctr += 1
            marker = f"no_imp ({patience_ctr})"

        print(
            f"Ep {epoch:03d} | loss={mean_train_loss:.4f} | "
            f"{_format_metrics(val_metrics)} | {marker}"
        )

        if patience_ctr >= patience:
            print("Early stopping triggered.")
            break


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_subject", type=int, default=1)

    parser.add_argument(
        "--seq_model",
        type=str,
        default="lstm",
        choices=["lstm", "rnn", "transformer"],
    )

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lb_lambda", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=40)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    dataset = DEAPDataset(
        physio_csv=PHYSIO_CSV,
        audio_csv=AUDIO_CSV,
        mat_dir=MAT_DIR,
        normalize=True,
    )

    train_fold(
        dataset,
        test_subject=args.test_subject,
        val_subject=(args.test_subject % 32) + 1,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        lb_lambda=args.lb_lambda,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        device=device,
        seq_model=args.seq_model,
    )


if __name__ == "__main__":
    main()