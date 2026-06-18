"""
data/combined_dataset.py

Unified DataLoader factory for training MoEEmotionModel.

Since WRIME (text) and RAVDESS (audio) are separate datasets with no
natural pairing, we train with modality-specific batches:
  - WRIME batches:   text inputs only   (audio_values = None)
  - RAVDESS batches: audio inputs only  (input_ids = None)

The MoEEmotionModel handles both cases gracefully via its text_available /
audio_available flags — the router and fusion layers degrade cleanly to
single-modality mode.

Usage:
    from data.combined_dataset import build_dataloaders
    loaders = build_dataloaders(config)
    for batch in loaders["train"]:
        outputs = model(**batch_to_model_inputs(batch))
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from typing import Dict, Optional
import numpy as np

from data.wrime_dataset   import WRIMEDataset
from data.ravdess_dataset import RAVDESSDataset
from data.emotion_mappings import NUM_CLASSES


# ── Collate functions ──────────────────────────────────────────────────────────

def _collate_wrime(batch):
    """Collate WRIME (text-only) samples. Audio fields are None."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"]  for b in batch]),
        "audio_values":   None,
        "label":          torch.stack([b["label"]           for b in batch]),
        "valence":        torch.stack([b["valence"]         for b in batch]),
        "arousal":        torch.stack([b["arousal"]         for b in batch]),
        "modality":       "text",
    }


def _collate_ravdess(batch):
    """Collate RAVDESS (audio-only) samples. Text fields are None."""
    return {
        "input_ids":      None,
        "attention_mask": None,
        "token_type_ids": None,
        "audio_values":   torch.stack([b["audio_values"] for b in batch]),
        "label":          torch.stack([b["label"]        for b in batch]),
        "valence":        torch.stack([b["valence"]      for b in batch]),
        "arousal":        torch.stack([b["arousal"]      for b in batch]),
        "modality":       "audio",
    }


# ── Weighted sampler for class imbalance ──────────────────────────────────────

def _make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples under-represented classes.
    Ensures each emotion class is seen roughly equally per epoch.
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)           # avoid division by zero
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_arr]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels_arr),
        replacement=True,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_dataloaders(
    config: Dict,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Build train/val/test DataLoaders for WRIME and RAVDESS.

    Args:
        config:      merged config dict (base.yaml + model.yaml + train.yaml)
        num_workers: DataLoader workers (set 0 for debugging)
        pin_memory:  pin memory for GPU transfer speed

    Returns:
        {
          "train": {"wrime": DataLoader, "ravdess": DataLoader},
          "val":   {"wrime": DataLoader, "ravdess": DataLoader},
          "test":  {"wrime": DataLoader, "ravdess": DataLoader},
        }

    Usage in training loop:
        for wrime_batch, ravdess_batch in zip(loaders["train"]["wrime"],
                                              loaders["train"]["ravdess"]):
            # process each batch independently — both call model.forward()
    """
    batch_size  = config["training"]["batch_size"]
    max_length  = config["text_encoder"].get("max_length", 128)
    model_name  = config["text_encoder"]["model_name"]
    ravdess_dir = config["data"].get("ravdess_path", "data/raw/ravdess")
    max_dur     = config["data"].get("max_audio_duration_s", 5.0)
    min_intens  = config["data"].get("wrime_min_intensity", 0.0)

    loaders: Dict[str, Dict[str, DataLoader]] = {}

    for split in ("train", "val", "test"):
        ravdess_split = "val" if split == "val" else split
        is_train = split == "train"

        wrime = WRIMEDataset(
            split=split if split != "val" else "validation",
            model_name=model_name,
            max_length=max_length,
            min_intensity=min_intens,
        )
        aug_cfg    = config.get("data", {}).get("augmentation", {}).get("audio", {})
        do_augment = aug_cfg.get("enabled", False) and is_train
        aug_p      = aug_cfg.get("p", 0.5)
        ravdess = RAVDESSDataset(
            data_dir=ravdess_dir,
            split=ravdess_split,
            max_duration_s=max_dur,
            augment=do_augment,
            augment_p=aug_p,
        )

        wrime_sampler = (
            _make_weighted_sampler([s["label"] for s in wrime.samples])
            if is_train else None
        )
        ravdess_sampler = (
            _make_weighted_sampler([s["label"] for s in ravdess.samples])
            if is_train else None
        )

        loaders[split] = {
            "wrime": DataLoader(
                wrime,
                batch_size=batch_size,
                sampler=wrime_sampler,
                shuffle=(is_train and wrime_sampler is None),
                collate_fn=_collate_wrime,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            "ravdess": DataLoader(
                ravdess,
                batch_size=batch_size,
                sampler=ravdess_sampler,
                shuffle=(is_train and ravdess_sampler is None),
                collate_fn=_collate_ravdess,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
        }

    return loaders


def batch_to_model_inputs(batch: Dict) -> Dict:
    """
    Extract only the fields that MoEEmotionModel.forward() expects.
    Filters out training-only fields (label, valence, arousal, modality).
    """
    return {
        k: v for k, v in batch.items()
        if k in {"input_ids", "attention_mask", "token_type_ids", "audio_values"}
        and v is not None
    }