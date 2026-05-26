"""
data/download_ravdess.py

Downloads the RAVDESS Audio Speech dataset from Zenodo and extracts it
into data/raw/ravdess/.

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
  - 24 actors (12F / 12M), English speech
  - 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
  - 2 intensity levels: normal, strong
  - ~1440 audio clips, ~24 actor folders
  - Zenodo DOI: 10.5281/zenodo.1188976

Usage:
    python data/download_ravdess.py
    python data/download_ravdess.py --output data/raw/ravdess
"""

import os
import sys
import zipfile
import argparse
import urllib.request
from pathlib import Path

ZENODO_URL = (
    "https://zenodo.org/record/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)
DEFAULT_OUTPUT = Path("data/raw/ravdess")
ZIP_NAME = "Audio_Speech_Actors_01-24.zip"


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    print(f"Downloading RAVDESS from Zenodo...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress


def extract(zip_path: Path, output_dir: Path) -> None:
    print(f"Extracting to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for i, member in enumerate(members):
            zf.extract(member, output_dir)
            pct = (i + 1) / len(members) * 100
            print(f"\r  Extracting {pct:5.1f}%  ({i+1}/{len(members)})", end="", flush=True)
    print()


def verify(output_dir: Path) -> bool:
    """Check at least 24 Actor folders and some .wav files exist."""
    actor_dirs = sorted(output_dir.glob("Actor_*"))
    wav_files  = list(output_dir.rglob("*.wav"))
    print(f"\nVerification:")
    print(f"  Actor folders : {len(actor_dirs)}")
    print(f"  WAV files     : {len(wav_files)}")
    ok = len(actor_dirs) >= 24 and len(wav_files) >= 1000
    print(f"  Status        : {'✓ OK' if ok else '✗ Incomplete — re-run download'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download RAVDESS audio dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the zip file after extraction")
    args = parser.parse_args()

    output_dir = args.output
    zip_path   = output_dir.parent / ZIP_NAME

    # Skip if already downloaded
    if output_dir.exists() and any(output_dir.glob("Actor_*")):
        print(f"RAVDESS already present at {output_dir}")
        verify(output_dir)
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Download
    download_with_progress(ZENODO_URL, zip_path)

    # Extract
    extract(zip_path, output_dir)

    # Cleanup zip
    if not args.keep_zip:
        zip_path.unlink()
        print(f"Removed {zip_path}")

    # Verify
    ok = verify(output_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()