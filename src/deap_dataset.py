# deap_dataset.py (FULLY FIXED)

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SUBJECTS = 32
N_TRIALS = 40

LABEL_COLS = ("valence", "arousal", "dominance", "liking")

PHYSIO_COLS = [
    f"ch{ch}_{stat}"
    for stat in ("mean", "std", "min", "max")
    for ch in range(1, 9)
]

AUDIO_COLS = ["note_density", "mean_pitch", "tempo"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DEAPDataset(Dataset):
    def __init__(
        self,
        physio_csv: str,
        audio_csv: str,
        mat_dir: Optional[str] = None,
        normalize: bool = True,
        transform=None,
    ):
        super().__init__()

        self.normalize = normalize
        self.transform = transform

        # Load data
        self._physio_df = self._load_physio(physio_csv)
        self._audio_df = self._load_audio(audio_csv)

        # Subject/trial mapping
        self.subject_ids = np.repeat(np.arange(1, N_SUBJECTS + 1), N_TRIALS)
        self.trial_ids = np.tile(np.arange(1, N_TRIALS + 1), N_SUBJECTS)

        # Tile audio (40 → 1280)
        audio_tiled = pd.concat([self._audio_df] * N_SUBJECTS, ignore_index=True)

        self._physio = self._physio_df[PHYSIO_COLS].values.astype(np.float32)
        self._audio = audio_tiled[AUDIO_COLS].values.astype(np.float32)

        # Load labels
        self._labels = self._load_labels(mat_dir)

        # Normalizer state
        self._physio_mean = None
        self._physio_std = None
        self._audio_mean = None
        self._audio_std = None

        # Debug sanity check (run once)
        print(f"[DEBUG] Labels range: {self._labels.min():.3f} to {self._labels.max():.3f}")

    # ------------------------------------------------------------------
    # Dataset methods
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._physio)

    def __getitem__(self, idx):
        physio = self._physio[idx].copy()
        audio = self._audio[idx].copy()

        # Normalize
        if self.normalize and self._physio_mean is not None:
            physio = (physio - self._physio_mean) / (self._physio_std + 1e-6)
            audio = (audio - self._audio_mean) / (self._audio_std + 1e-6)

        mask = np.array([1.0, 1.0], dtype=np.float32)

        physio = torch.from_numpy(physio)
        audio = torch.from_numpy(audio)
        mask = torch.from_numpy(mask)

        if self.transform:
            physio = self.transform(physio)

        label_row = self._labels[idx]

        targets = {
            col: torch.tensor([label_row[i]], dtype=torch.float32)
            for i, col in enumerate(LABEL_COLS)
        }

        return physio, audio, mask, targets

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def fit_normalizer(self, indices: List[int]):
        physio = self._physio[indices]
        audio = self._audio[indices]

        self._physio_mean = physio.mean(axis=0)
        self._physio_std = physio.std(axis=0) + 1e-6

        self._audio_mean = audio.mean(axis=0)
        self._audio_std = audio.std(axis=0) + 1e-6

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_physio(path):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def _load_audio(path):
        df = pd.read_csv(path)

        # Sort by trial number
        if "file" in df.columns:
            df = df.copy()
            df["_trial"] = df["file"].str.extract(r"exp_id_(\d+)").astype(int)
            df = df.sort_values("_trial").drop(columns=["_trial", "file"])

        return df.reset_index(drop=True)

    @staticmethod
    def _load_labels(mat_dir):
        if mat_dir is None:
            warnings.warn("Using dummy labels")
            return np.zeros((N_SUBJECTS * N_TRIALS, 4), dtype=np.float32)

        import pickle

        all_labels = []

        for subj in range(1, N_SUBJECTS + 1):
            path = Path(mat_dir) / f"s{subj:02d}.dat"

            with open(path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            labels = np.array(data["labels"], dtype=np.float32)

            # 🔥 CRITICAL FIX: Normalize labels
            labels = (labels - 5.0) / 4.0

            all_labels.append(labels)

        return np.vstack(all_labels)


# ---------------------------------------------------------------------------
# LOSO Split
# ---------------------------------------------------------------------------

def loso_split(dataset, test_subject, val_subject=None):
    subject_ids = dataset.subject_ids

    test_idx = np.where(subject_ids == test_subject)[0].tolist()
    val_idx = np.where(subject_ids == val_subject)[0].tolist() if val_subject else []

    train_idx = np.where(
        (subject_ids != test_subject) &
        (subject_ids != (val_subject or -1))
    )[0].tolist()

    return {
        "train": Subset(dataset, train_idx),
        "val": Subset(dataset, val_idx),
        "test": Subset(dataset, test_idx),
    }