"""
data/wrime_dataset.py

PyTorch Dataset for WRIME — Japanese text emotion corpus.

WRIME (Word Reading In Multiple Emotions):
  - ~43,000 Japanese social media posts
  - Annotated with Plutchik emotion intensities (0-3) by writer + readers
  - Source: https://github.com/ids-cv/wrime
  - Downloaded automatically to data/raw/wrime/ on first run

Emotion mapping: Plutchik 8 → our 7 classes (see data/emotion_mappings.py)
Valence/Arousal: derived from dominant emotion + intensity via Russell circumplex
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ssl
import urllib.request
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl._create_unverified_context()

from data.emotion_mappings import (
    EMOTION_TO_IDX,
    WRIME_TO_EMOTION,
    WRIME_INTENSITY_SCALE,
    emotion_to_va,
)

# Git LFS media URL — correct path for LFS-stored files in public GitHub repos
WRIME_URLS = [
    "https://media.githubusercontent.com/media/ids-cv/wrime/master/wrime-ver1.tsv",
    "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver1.tsv",
]
WRIME_PATH = Path("data/raw/wrime/wrime-ver1.tsv")

WRITER_COLS = {
    "Writer_Joy":          "joy",
    "Writer_Sadness":      "sadness",
    "Writer_Anticipation": "anticipation",
    "Writer_Surprise":     "surprise",
    "Writer_Anger":        "anger",
    "Writer_Fear":         "fear",
    "Writer_Disgust":      "disgust",
    "Writer_Trust":        "trust",
}


def _download_wrime() -> None:
    """Download wrime-ver1.tsv from GitHub (LFS media URL) if not already present."""
    if WRIME_PATH.exists() and WRIME_PATH.stat().st_size > 1_000_000:
        return  # already downloaded and looks complete (>1MB)
    WRIME_PATH.parent.mkdir(parents=True, exist_ok=True)

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=SSL_CONTEXT)
    )
    urllib.request.install_opener(opener)

    for url in WRIME_URLS:
        print(f"Downloading WRIME...\n  URL: {url}")
        try:
            def reporthook(b, bs, total):
                if total > 0:
                    pct = min(b * bs / total * 100, 100)
                    mb  = min(b * bs, total) / 1e6
                    print(f"\r  {pct:5.1f}%  {mb:.1f}/{total/1e6:.1f} MB",
                          end="", flush=True)

            urllib.request.urlretrieve(url, WRIME_PATH, reporthook=reporthook)
            print()

            # Verify it's a real TSV (LFS pointer files are ~130 bytes)
            if WRIME_PATH.stat().st_size < 1_000_000:
                print("  Got LFS pointer, not the real file. Trying next URL...")
                WRIME_PATH.unlink()
                continue

            print(f"  Saved to {WRIME_PATH} "
                  f"({WRIME_PATH.stat().st_size / 1e6:.1f} MB)")
            return

        except Exception as e:
            print(f"  Failed: {e}")
            if WRIME_PATH.exists():
                WRIME_PATH.unlink()
            continue

    # All URLs failed — give manual instructions
    print("\n" + "="*60)
    print("WRIME automatic download failed.")
    print("Please download manually:")
    print("  1. Go to: https://github.com/ids-cv/wrime")
    print("  2. Click 'wrime-ver1.tsv' → 'View raw' → save file")
    print(f"  3. Place at: {WRIME_PATH.resolve()}")
    print("="*60 + "\n")
    raise RuntimeError("WRIME download failed. See instructions above.")


def _dominant_emotion(row: pd.Series) -> Tuple[str, float]:
    scores = {
        plutchik: float(row[col])
        for col, plutchik in WRITER_COLS.items()
        if col in row.index
    }
    dominant  = max(scores, key=scores.get)
    intensity = scores[dominant]
    if intensity == 0:
        return "trust", 0.0   # maps to neutral at zero intensity
    return dominant, intensity / WRIME_INTENSITY_SCALE


def _make_splits(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n  = len(df)
    t  = int(n * 0.70)
    v  = int(n * 0.85)
    return {
        "train":      df.iloc[:t],
        "validation": df.iloc[t:v],
        "test":       df.iloc[v:],
    }


class WRIMEDataset(Dataset):
    """
    Tokenised WRIME dataset for text-only emotion classification.

    Each sample:
        input_ids       (seq_len,)
        attention_mask  (seq_len,)
        token_type_ids  (seq_len,)
        label           ()   int64 — emotion class index
        valence         ()   float32 in [1, 9]
        arousal         ()   float32 in [1, 9]
    """

    def __init__(
        self,
        split: str = "train",
        model_name: str = "cl-tohoku/bert-base-japanese-v3",
        max_length: int = 128,
        min_intensity: float = 0.0,
        data_path: str = str(WRIME_PATH),
    ):
        super().__init__()
        self.max_length    = max_length
        self.min_intensity = min_intensity

        _download_wrime()

        print(f"Loading WRIME ({split})...")
        df = pd.read_csv(data_path, sep="\t")
        df.columns = [c.strip() for c in df.columns]

        splits = _make_splits(df)
        if split not in splits:
            raise ValueError(f"split must be train/validation/test, got '{split}'")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.samples   = self._process(splits[split])
        print(f"  {len(self.samples)} samples (split={split})")

    def _process(self, df: pd.DataFrame) -> List[Dict]:
        samples  = []
        text_col = "Sentence" if "Sentence" in df.columns else df.columns[0]

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text or text == "nan":
                continue

            plutchik, intensity = _dominant_emotion(row)
            if intensity < self.min_intensity:
                continue

            emotion          = WRIME_TO_EMOTION[plutchik]
            label            = EMOTION_TO_IDX[emotion]
            valence, arousal = emotion_to_va(emotion, intensity)

            samples.append({
                "sentence": text,
                "label":    label,
                "valence":  valence,
                "arousal":  arousal,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s   = self.samples[idx]
        enc = self.tokenizer(
            s["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            "label":   torch.tensor(s["label"],   dtype=torch.long),
            "valence": torch.tensor(s["valence"], dtype=torch.float32),
            "arousal": torch.tensor(s["arousal"], dtype=torch.float32),
        }

    def label_distribution(self) -> Dict[str, int]:
        from collections import Counter
        from data.emotion_mappings import EMOTION_CLASSES
        counts = Counter(s["label"] for s in self.samples)
        return {EMOTION_CLASSES[i]: counts.get(i, 0) for i in range(len(EMOTION_CLASSES))}