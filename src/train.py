# train.py (FINAL FIXED - NO IMPORT ERRORS)

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deap_dataset import DEAPDataset, loso_split
from deap_emotion_model import DEAPEmotionModel

PHYSIO_CSV  = str(SRC_DIR / "data" / "physio_features.csv")
AUDIO_CSV   = str(SRC_DIR / "data" / "audio_features.csv")
MAT_DIR     = str(SRC_DIR.parent / "db" / "deap-dataset" / "data_preprocessed_python")
CKPT_DIR    = str(SRC_DIR / "checkpoints")
RUNS_DIR    = str(SRC_DIR.parent / "runs")

DIMENSIONS  = DEAPEmotionModel.DIMENSIONS


# ===========================================================================
# Helpers
# ===========================================================================

def _aggregate_val_outputs(val_outputs):
    agg = {"mean_rmse": [], "mean_ccc": []}
    for dim in DIMENSIONS:
        agg[f"rmse_{dim}"] = []
        agg[f"ccc_{dim}"]  = []

    for out in val_outputs:
        agg["mean_rmse"].append(out.mean_rmse)
        agg["mean_ccc"].append(out.mean_ccc)
        for dim in DIMENSIONS:
            agg[f"rmse_{dim}"].append(out.rmse[dim])
            agg[f"ccc_{dim}"].append(out.ccc[dim])

    return {k: float(np.mean(v)) for k, v in agg.items()}


def _format_metrics(metrics):
    parts = [
        f"CCC={metrics['mean_ccc']:.4f}",
        f"RMSE={metrics['mean_rmse']:.4f}",
    ]
    for dim in DIMENSIONS:
        parts.append(f"{dim[:3]}_ccc={metrics[f'ccc_{dim}']:.3f}")
    return "  ".join(parts)


# ===========================================================================
# Training
# ===========================================================================

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
):

    print(f"\n{'='*60}")
    print(f"FOLD {test_subject} | val={val_subject}")
    print(f"{'='*60}")

    splits = loso_split(dataset, test_subject, val_subject)
    dataset.fit_normalizer(splits["train"].indices)

    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(splits["val"], batch_size=batch_size, shuffle=False)

    model = DEAPEmotionModel(embed_dim=embed_dim, dropout=dropout, lb_lambda=lb_lambda).to(device)
    optimizer, scheduler = model.configure_optimizer(lr=lr, weight_decay=weight_decay, t_max=max_epochs)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_ccc = -1e9
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):

        # ---------------- TRAIN ----------------
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

        # ---------------- VALIDATION ----------------
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

        # ---------------- EARLY STOP ----------------
        if val_ccc > best_val_ccc + 1e-3:
            best_val_ccc = val_ccc
            patience_ctr = 0
            os.makedirs(CKPT_DIR, exist_ok=True)
            model.save(os.path.join(CKPT_DIR, f"fold_{test_subject:02d}.pt"))
            marker = "✓"
        else:
            patience_ctr += 1
            marker = f"no_imp {patience_ctr}"

        print(f"Ep {epoch:3d} | loss={mean_train_loss:.4f} | {_format_metrics(val_metrics)} | {marker}")

        if patience_ctr >= patience:
            print("Early stopping.")
            break


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_subject", type=int, default=1)

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lb_lambda", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    )


if __name__ == "__main__":
    main()