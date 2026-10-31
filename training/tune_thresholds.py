"""
training/tune_thresholds.py

Per-class decision threshold tuning for the trained MoEEmotionModel.

After training, the default argmax (equivalent to equal 1/7 threshold per class)
is suboptimal for imbalanced data. This script sweeps thresholds on the
validation set and saves the best per-class thresholds to a JSON file.

At inference time, EmotionPipeline reads this file and applies the thresholds
instead of argmax. Typical improvement: +0.03–0.06 macro-F1 at zero training cost.

Usage:
    python training/tune_thresholds.py
    python training/tune_thresholds.py --checkpoint outputs/checkpoints/best_p2.pt
    python training/tune_thresholds.py --out outputs/thresholds.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

from data.cached_dataset import build_cached_dataloaders, FEATURE_DIR
from data.combined_dataset import build_dataloaders
from data.emotion_mappings import EMOTION_CLASSES, NUM_CLASSES
from models.multimodal.moe import MoEEmotionModel


DEFAULT_CKPT = "outputs/checkpoints/best.pt"
DEFAULT_OUT  = "outputs/thresholds.json"
THRESHOLD_GRID = np.arange(0.05, 0.80, 0.025)


def load_config(config_dir: str = "configs/") -> dict:
    merged = {}
    for name in ("base", "model", "train"):
        p = Path(config_dir) / f"{name}.yaml"
        if p.exists():
            with open(p) as f:
                merged.update(yaml.safe_load(f))
    return merged


def collect_val_probs(model, loaders, device) -> tuple:
    """Run the model over the validation set and collect probs + labels."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for mod in ("wrime", "ravdess"):
            for batch in tqdm(loaders["val"][mod], desc=f"val/{mod}", leave=False):
                def to(x):
                    return x.to(device) if x is not None else None

                if "text_features" in batch:
                    kwargs = {
                        "text_features":  to(batch["text_features"]),
                        "audio_features": to(batch["audio_features"]),
                    }
                else:
                    kwargs = {
                        k: to(v)
                        for k, v in batch.items()
                        if k in {"input_ids", "attention_mask",
                                 "token_type_ids", "audio_values"}
                        and v is not None
                    }

                out    = model(**kwargs)
                probs  = out["probs"].cpu().numpy()      # (B, 7)
                labels = batch["label"].numpy()          # (B,)
                all_probs.append(probs)
                all_labels.append(labels)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def tune_thresholds(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    For each class, sweep a threshold and pick the value that maximises
    that class's F1 score on the validation set.

    Returns a dict mapping class name → best threshold (float).
    """
    best_thresholds = {}

    for cls_idx, cls_name in enumerate(EMOTION_CLASSES):
        binary_labels = (labels == cls_idx).astype(int)
        cls_probs     = probs[:, cls_idx]
        best_f1, best_t = 0.0, 0.5

        for t in THRESHOLD_GRID:
            preds = (cls_probs >= t).astype(int)
            f1    = f1_score(binary_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        best_thresholds[cls_name] = round(best_t, 4)
        print(f"  {cls_name:12s}  best_threshold={best_t:.3f}  F1={best_f1:.3f}")

    return best_thresholds


def evaluate_with_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: dict,
) -> float:
    """Apply per-class thresholds and return macro-F1 on the validation set."""
    threshold_arr = np.array([thresholds[c] for c in EMOTION_CLASSES])
    # For each sample: class with highest (prob - threshold) above 0, else argmax
    adjusted = probs - threshold_arr[np.newaxis, :]
    preds    = adjusted.argmax(axis=1)
    return f1_score(labels, preds, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--config",     default="configs/")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--out",        default=DEFAULT_OUT)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    model = MoEEmotionModel(config).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build dataloaders — prefer cached features for speed
    cache_ok = all(
        (FEATURE_DIR / f"{ds}_{sp}.pt").exists()
        for ds in ("wrime", "ravdess")
        for sp in ("train", "validation", "test")
    )
    if cache_ok:
        print("Using cached features.")
        loaders = build_cached_dataloaders(batch_size=512)
    else:
        print("Using raw dataloaders (slower).")
        loaders = build_dataloaders(config, num_workers=0)

    # Collect validation probabilities
    print("\nCollecting validation probabilities...")
    probs, labels = collect_val_probs(model, loaders, device)
    print(f"Validation samples: {len(labels)}")

    # Baseline: argmax
    baseline_preds = probs.argmax(axis=1)
    baseline_f1    = f1_score(labels, baseline_preds, average="macro", zero_division=0)
    print(f"\nBaseline macro-F1 (argmax): {baseline_f1:.4f}")

    # Tune per-class thresholds
    print("\nTuning per-class thresholds...")
    thresholds = tune_thresholds(probs, labels)

    # Evaluate with tuned thresholds
    tuned_f1 = evaluate_with_thresholds(probs, labels, thresholds)
    print(f"\nTuned macro-F1 (thresholds): {tuned_f1:.4f}  (Δ {tuned_f1 - baseline_f1:+.4f})")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nThresholds saved → {out_path}")
    print("Load in EmotionPipeline with: pipe.load_thresholds(path)")


if __name__ == "__main__":
    main()
