"""api/routes/audio_file.py — POST /predict/file"""

import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional

from api.schemas import EmotionResponse
from api.dependencies import get_pipeline
from inference.pipeline import EmotionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])

MAX_FILE_SIZE = 30 * 1024 * 1024   # 30 MB


@router.post("/file", response_model=EmotionResponse)
async def predict_file(
    audio: UploadFile = File(
        ...,
        description="Audio file. Supported: wav, flac, mp3, m4a, ogg. Max 30s.",
    ),
    text: Optional[str] = Form(
        None,
        description="Optional Japanese text to combine with audio (multimodal).",
    ),
    pipe: EmotionPipeline = Depends(get_pipeline),
):
    """
    Predict emotion from an uploaded audio file.

    Accepts wav, flac, mp3, m4a, ogg. Handles decoding, mono mixing, and
    resampling to 16kHz internally. Optionally combine with Japanese text
    for multimodal prediction.
    """
    # Validate content type
    allowed = {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/flac", "audio/x-flac",
        "audio/mpeg", "audio/mp3",
        "audio/mp4", "audio/m4a", "audio/x-m4a",
        "audio/ogg", "application/ogg",
        "application/octet-stream",   # generic binary — allow and let decoder decide
    }
    ct = (audio.content_type or "").lower()
    if ct and ct not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {ct}. "
                   "Use wav, flac, mp3, m4a, or ogg.",
        )

    # Read bytes
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(audio_bytes)/1e6:.1f} MB). Max 30 MB.",
        )

    # Infer format from filename extension as hint
    fmt = None
    if audio.filename:
        ext = audio.filename.rsplit(".", 1)[-1].lower()
        if ext in ("wav", "flac", "mp3", "m4a", "ogg"):
            fmt = ext

    try:
        if text:
            result = pipe.predict(text=text.strip(), audio_bytes=audio_bytes, fmt=fmt)
        else:
            result = pipe.predict_audio_bytes(audio_bytes, fmt=fmt)
        return EmotionResponse(**result.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("File prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")