import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
training/cache_features.py

Pre-computes and saves encoder features for all WRIME and RAVDESS splits.

Run this ONCE before training:
    python training/cache_features.py

What it does:
  - Passes all WRIME text through the frozen BERT encoder  → saves 768-dim vectors
  - Passes all RAVDESS audio through the frozen Wav2Vec2   → saves 768-dim vectors
  - Saves to data/raw/features/{wrime,ravdess}_{split}.pt

After caching, train.py skips the encoders entirely each step — only
router + fusion + heads are trained. Epoch time drops from ~8min to <1min.

Cache size: ~140 MB total (manageable).
Cache time: ~10 min one-time cost (BERT + Wav2Vec2 forward passes only).

Re-run only if you change encoder model_name or max_length/max_duration in configs.
"""

import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.wrime_dataset    import WRIMEDataset
from data.ravdess_dataset  import RAVDESSDataset
from models.text.encoder   import JapaneseTextEncoder
from models.audio.encoder  import AudioEncoder

FEATURE_DIR = Path("data/raw/features")


def cache_wrime(encoder: JapaneseTextEncoder, config: dict, device: torch.device):
    """Encode all WRIME splits and save features to disk."""
    max_length  = config["text_encoder"].get("max_length", 64)
    model_name  = config["text_encoder"]["model_name"]
    batch_size  = 128   # larger batch for fast encoding (no backward needed)

    for split in ("train", "validation", "test"):
        out_path = FEATURE_DIR / f"wrime_{split}.pt"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue

        ds = WRIMEDataset(split=split, model_name=model_name, max_length=max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

        all_features, all_labels, all_valence, all_arousal = [], [], [], []

        print(f"  Encoding WRIME {split} ({len(ds)} samples)...")
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"  wrime/{split}", leave=False):
                tti = batch.get("token_type_ids")
                out = encoder(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    token_type_ids = tti.to(device) if tti is not None else None,
                )
                all_features.append(out["features"].cpu())
                all_labels.append(batch["label"])
                all_valence.append(batch["valence"])
                all_arousal.append(batch["arousal"])

        torch.save({
            "features": torch.cat(all_features),   # (N, 768)
            "labels":   torch.cat(all_labels),     # (N,)
            "valence":  torch.cat(all_valence),    # (N,)
            "arousal":  torch.cat(all_arousal),    # (N,)
        }, out_path)

        mb = out_path.stat().st_size / 1e6
        print(f"  Saved {out_path}  ({mb:.1f} MB, {len(ds)} samples)")


def cache_ravdess(encoder: AudioEncoder, config: dict, device: torch.device):
    """Encode all RAVDESS splits and save features to disk."""
    ravdess_dir = config["data"].get("ravdess_path", "data/raw/ravdess")
    max_dur     = config["data"].get("max_audio_duration_s", 3.0)
    batch_size  = 32    # smaller; audio tensors are large

    split_map = {"train": "train", "validation": "val", "test": "test"}

    for hf_split, rav_split in split_map.items():
        out_path = FEATURE_DIR / f"ravdess_{hf_split}.pt"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue

        ds = RAVDESSDataset(data_dir=ravdess_dir, split=rav_split,
                            max_duration_s=max_dur)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

        all_features, all_labels, all_valence, all_arousal = [], [], [], []

        print(f"  Encoding RAVDESS {rav_split} ({len(ds)} clips)...")
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"  ravdess/{rav_split}", leave=False):
                out = encoder(audio_values=batch["audio_values"].to(device))
                all_features.append(out["features"].cpu())
                all_labels.append(batch["label"])
                all_valence.append(batch["valence"])
                all_arousal.append(batch["arousal"])

        torch.save({
            "features": torch.cat(all_features),
            "labels":   torch.cat(all_labels),
            "valence":  torch.cat(all_valence),
            "arousal":  torch.cat(all_arousal),
        }, out_path)

        mb = out_path.stat().st_size / 1e6
        print(f"  Saved {out_path}  ({mb:.1f} MB, {len(ds)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config
    config_dir = Path(args.config)
    config = {}
    for name in ("base", "model", "train"):
        with open(config_dir / f"{name}.yaml") as f:
            config.update(yaml.safe_load(f))

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Text encoder (frozen) ──────────────────────────────────────────────────
    print("\nLoading text encoder...")
    text_encoder = JapaneseTextEncoder(config["text_encoder"]).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    print("\nCaching WRIME features...")
    cache_wrime(text_encoder, config, device)
    del text_encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    # ── Audio encoder (frozen) ─────────────────────────────────────────────────
    print("\nLoading audio encoder...")
    audio_encoder = AudioEncoder(config["audio_encoder"]).to(device)
    audio_encoder.eval()
    for p in audio_encoder.parameters():
        p.requires_grad = False

    print("\nCaching RAVDESS features...")
    cache_ravdess(audio_encoder, config, device)
    del audio_encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    print("\nDone. All features cached to data/raw/features/")
    print("Now run: python training/train.py")


if __name__ == "__main__":
    main()