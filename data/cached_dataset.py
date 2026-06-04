import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
data/cached_dataset.py

Lightweight Dataset loading pre-computed encoder features from disk.
Used by train.py after cache_features.py has been run.

Changes vs original:
  - WeightedRandomSampler on WRIME train loader so every emotion class
    is seen equally often regardless of raw frequency.
  - RAVDESS train loader also sampled by class weight (small dataset,
    class imbalance still present even after RAVDESS-internal split).
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict

from data.emotion_mappings import NUM_CLASSES

FEATURE_DIR = Path("data/raw/features")


class CachedDataset(Dataset):
    def __init__(self, path: Path, modality: str):
        if not path.exists():
            raise FileNotFoundError(
                f"Feature cache not found: {path}\n"
                "Run: python training/cache_features.py"
            )
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.features = data["features"]
        self.labels   = data["labels"]
        self.valence  = data["valence"]
        self.arousal  = data["arousal"]
        self.modality = modality
        print(f"  Loaded {path.name}: {len(self)} samples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label":    self.labels[idx],
            "valence":  self.valence[idx],
            "arousal":  self.arousal[idx],
        }


def _make_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """
    Returns a sampler that draws each class with equal probability.
    Minority classes (angry, anxious, surprised, disgusted) get upsampled;
    majority classes (happy, sad, neutral) get downsampled accordingly.
    """
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    weight_per_class = 1.0 / counts.clamp(min=1)           # inverse frequency
    sample_weights   = weight_per_class[labels]             # one weight per sample
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(labels),   # same epoch length as before
        replacement = True,
    )


def _collate_text(batch):
    return {
        "text_features":  torch.stack([b["features"] for b in batch]),
        "audio_features": None,
        "label":   torch.stack([b["label"]   for b in batch]),
        "valence": torch.stack([b["valence"] for b in batch]),
        "arousal": torch.stack([b["arousal"] for b in batch]),
        "modality": "text",
    }


def _collate_audio(batch):
    return {
        "text_features":  None,
        "audio_features": torch.stack([b["features"] for b in batch]),
        "label":   torch.stack([b["label"]   for b in batch]),
        "valence": torch.stack([b["valence"] for b in batch]),
        "arousal": torch.stack([b["arousal"] for b in batch]),
        "modality": "audio",
    }


def build_cached_dataloaders(batch_size: int = 256) -> Dict:
    """
    Build DataLoaders from cached feature files.
    batch_size can be large (no encoder forward pass per step).
    Returns: {"train": {"wrime": DL, "ravdess": DL}, "val": {...}, "test": {...}}

    Training loaders use WeightedRandomSampler for balanced class exposure.
    Val/test loaders are unshuffled and unweighted (true distribution).
    """
    loaders = {}
    splits = {
        "train": ("train",      True),
        "val":   ("validation", False),
        "test":  ("test",       False),
    }
    for split, (file_split, is_train) in splits.items():
        wrime_ds   = CachedDataset(FEATURE_DIR / f"wrime_{file_split}.pt",   "text")
        ravdess_ds = CachedDataset(FEATURE_DIR / f"ravdess_{file_split}.pt", "audio")

        if is_train:
            wrime_sampler   = _make_weighted_sampler(wrime_ds.labels)
            ravdess_sampler = _make_weighted_sampler(ravdess_ds.labels)
            wrime_loader = DataLoader(
                wrime_ds, batch_size=batch_size,
                sampler=wrime_sampler,          # replaces shuffle=True
                num_workers=0, pin_memory=False,
                collate_fn=_collate_text,
            )
            ravdess_loader = DataLoader(
                ravdess_ds, batch_size=batch_size,
                sampler=ravdess_sampler,
                num_workers=0, pin_memory=False,
                collate_fn=_collate_audio,
            )
        else:
            wrime_loader = DataLoader(
                wrime_ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
                collate_fn=_collate_text,
            )
            ravdess_loader = DataLoader(
                ravdess_ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
                collate_fn=_collate_audio,
            )

        loaders[split] = {"wrime": wrime_loader, "ravdess": ravdess_loader}

    return loaders