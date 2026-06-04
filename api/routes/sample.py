"""
api/routes/sample.py

GET /sample

Serves a real RAVDESS audio clip so the frontend "Try a sample" button
works without the user needing their own audio file.

Returns a random clip each call, cycling through a small curated set that
covers different emotions so repeated clicks give varied results.
"""

import random
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["sample"])

RAVDESS_DIR = Path("data/raw/ravdess")

# Curated set: one clip per emotion, normal intensity, Actor_01
# Filename: modality-vocal-emotion-intensity-statement-rep-actor.wav
# emotion codes: 01=neutral, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
SAMPLE_CLIPS = [
    "03-01-03-01-01-01-01.wav",   # happy
    "03-01-04-01-01-01-01.wav",   # sad
    "03-01-05-01-01-01-01.wav",   # angry
    "03-01-06-01-01-01-01.wav",   # fearful → anxious
    "03-01-07-01-01-01-01.wav",   # disgust
    "03-01-08-01-01-01-01.wav",   # surprised
    "03-01-01-01-01-01-01.wav",   # neutral
]

EMOTION_NAMES = {
    "03": "happy", "04": "sad", "05": "angry",
    "06": "anxious", "07": "disgusted", "08": "surprised", "01": "neutral",
}


def _find_clip(filename: str) -> Path | None:
    """Search all Actor folders for the given filename."""
    for actor_dir in sorted(RAVDESS_DIR.glob("Actor_*")):
        candidate = actor_dir / filename
        if candidate.exists():
            return candidate
    return None


def _any_wav() -> Path | None:
    """Fallback: return any wav file found in the RAVDESS directory."""
    wavs = list(RAVDESS_DIR.rglob("*.wav"))
    return random.choice(wavs) if wavs else None


@router.get("/sample")
async def get_sample():
    """
    Returns a random RAVDESS audio clip as audio/wav.

    The frontend receives raw bytes, wraps them in a File object,
    and loads them into the file upload dropzone — ready to predict.

    Cycles through one clip per emotion class for variety.
    """
    if not RAVDESS_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="RAVDESS audio data not found. Run: python data/download_ravdess.py",
        )

    # Try curated clips first
    random.shuffle(SAMPLE_CLIPS)
    for filename in SAMPLE_CLIPS:
        path = _find_clip(filename)
        if path:
            emotion_code = filename.split("-")[2]
            emotion = EMOTION_NAMES.get(emotion_code, "unknown")
            return FileResponse(
                path=str(path),
                media_type="audio/wav",
                filename=f"sample_{emotion}.wav",
                headers={"X-Emotion-Hint": emotion},
            )

    # Fallback: any wav in the directory
    fallback = _any_wav()
    if fallback:
        return FileResponse(
            path=str(fallback),
            media_type="audio/wav",
            filename="sample.wav",
        )

    raise HTTPException(
        status_code=404,
        detail="No audio samples found. Run: python data/download_ravdess.py",
    )