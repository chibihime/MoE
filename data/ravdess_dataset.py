"""
data/ravdess_dataset.py

PyTorch Dataset for RAVDESS — English emotional speech audio.

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):
  - 24 professional actors (12F / 12M)
  - 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
  - 2 intensity levels: normal (01), strong (02)
  - ~1440 audio clips, 16-bit 48kHz WAV → resampled to 16kHz
  - Download: python data/download_ravdess.py

Filename format:  {modality}-{vocal}-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav
Example:          03-01-05-02-01-01-12.wav
  modality 03 = audio-only speech
  emotion  05 = angry
  intensity 02 = strong

Uses soundfile for loading (avoids torchaudio torchcodec dependency).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import torch
import torchaudio.functional as F_audio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from data.emotion_mappings import (
    EMOTION_TO_IDX,
    RAVDESS_CODE_TO_EMOTION,
    RAVDESS_INTENSITY_MAP,
    emotion_to_va,
)

TARGET_SAMPLE_RATE = 16_000
FILENAME_PATTERN   = re.compile(
    r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$"
)


def _parse_filename(path: Path) -> Optional[Dict]:
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    modality, vocal, emotion_code, intensity_code, statement, rep, actor = \
        [int(x) for x in m.groups()]

    if modality != 3:
        return None
    if emotion_code not in RAVDESS_CODE_TO_EMOTION:
        return None

    emotion   = RAVDESS_CODE_TO_EMOTION[emotion_code]
    intensity = RAVDESS_INTENSITY_MAP.get(intensity_code, 0.6)
    valence, arousal = emotion_to_va(emotion, intensity)

    return {
        "path":      path,
        "emotion":   emotion,
        "label":     EMOTION_TO_IDX[emotion],
        "intensity": intensity,
        "valence":   valence,
        "arousal":   arousal,
        "actor":     actor,
        "native_sr": 48_000,   # RAVDESS is always 48kHz
    }


class RAVDESSDataset(Dataset):
    """
    Audio-only dataset from RAVDESS for emotion classification.

    Each sample:
        audio_values  (num_samples,)  — float32 waveform at 16kHz
        label         ()              — int64 emotion class index
        valence       ()              — float32 in [1, 9]
        arousal       ()              — float32 in [1, 9]
    """

    def __init__(
        self,
        data_dir: str = "data/raw/ravdess",
        split: str = "train",
        max_duration_s: float = 5.0,
        train_actors: Optional[List[int]] = None,
        val_actors:   Optional[List[int]] = None,
        test_actors:  Optional[List[int]] = None,
    ):
        super().__init__()
        self.max_samples  = int(max_duration_s * TARGET_SAMPLE_RATE)
        self.target_sr    = TARGET_SAMPLE_RATE

        train_actors = train_actors or list(range(1, 19))
        val_actors   = val_actors   or [19, 20, 21]
        test_actors  = test_actors  or [22, 23, 24]

        split_map = {"train": train_actors, "val": val_actors, "test": test_actors}
        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map)}; got '{split}'")
        allowed_actors = set(split_map[split])

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"RAVDESS not found at {data_dir}.\n"
                "Run: python data/download_ravdess.py"
            )

        print(f"Scanning RAVDESS ({split}) at {data_dir} ...")
        self.samples: List[Dict] = []
        for wav in sorted(data_path.rglob("*.wav")):
            meta = _parse_filename(wav)
            if meta is None:
                continue
            if meta["actor"] not in allowed_actors:
                continue
            self.samples.append(meta)

        print(f"  {len(self.samples)} clips loaded (split={split}, "
              f"actors={sorted(allowed_actors)})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # Load with soundfile — avoids torchaudio torchcodec dependency
        waveform_np, sr = sf.read(str(s["path"]), dtype="float32", always_2d=True)
        # waveform_np: (samples, channels) → transpose to (channels, samples)
        waveform = torch.from_numpy(waveform_np.T)  # (channels, samples)

        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (samples,)

        # Resample to 16kHz using torchaudio.functional (no torchcodec needed)
        if sr != self.target_sr:
            waveform = F_audio.resample(
                waveform.unsqueeze(0),
                orig_freq=sr,
                new_freq=self.target_sr,
            ).squeeze(0)

        # Pad or truncate to fixed length
        n = waveform.shape[0]
        if n < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - n))
        else:
            waveform = waveform[:self.max_samples]

        return {
            "audio_values": waveform,
            "label":        torch.tensor(s["label"],   dtype=torch.long),
            "valence":      torch.tensor(s["valence"], dtype=torch.float32),
            "arousal":      torch.tensor(s["arousal"], dtype=torch.float32),
        }

    def label_distribution(self) -> Dict[str, int]:
        from collections import Counter
        from data.emotion_mappings import EMOTION_CLASSES
        counts = Counter(s["label"] for s in self.samples)
        return {EMOTION_CLASSES[i]: counts.get(i, 0) for i in range(len(EMOTION_CLASSES))}