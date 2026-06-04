"""
api/schemas.py

Pydantic request and response schemas.

Keeping these in one file means the API contract is visible in one place
and auto-documented by FastAPI's OpenAPI generation.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


# ── Requests ───────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Japanese text to classify.",
        examples=["今日はとても嬉しい！"],
    )


class StreamChunkRequest(BaseModel):
    """
    Metadata sent alongside a PCM stream chunk over WebSocket.
    The raw audio bytes are sent as binary frames; this comes as a JSON frame.
    """
    text: Optional[str] = Field(
        None,
        description="Optional concurrent text from live ASR.",
    )
    sample_rate: int = Field(
        16000,
        description="Sample rate of the PCM chunk. Must be 16000.",
    )


# ── Responses ──────────────────────────────────────────────────────────────────

class EmotionResponse(BaseModel):
    emotion:      str   = Field(..., description="Predicted emotion label.")
    confidence:   float = Field(..., description="Probability of predicted class (0–1).")
    valence:      float = Field(..., description="Valence score (1–9).")
    arousal:      float = Field(..., description="Arousal score (1–9).")
    all_probs:    Dict[str, float] = Field(
        ..., description="Softmax probability for each of the 7 emotion classes."
    )
    modality:     str   = Field(..., description="text | audio | multimodal")
    text_weight:  Optional[float] = Field(None, description="Router weight for text expert.")
    audio_weight: Optional[float] = Field(None, description="Router weight for audio expert.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "emotion": "sad",
                "confidence": 0.51,
                "valence": 2.8,
                "arousal": 3.1,
                "all_probs": {
                    "happy": 0.05,
                    "sad": 0.51,
                    "angry": 0.08,
                    "neutral": 0.14,
                    "anxious": 0.12,
                    "surprised": 0.06,
                    "disgusted": 0.04,
                },
                "modality": "text",
                "text_weight": 1.0,
                "audio_weight": 0.0,
            }
        }
    }


class StreamResponse(BaseModel):
    """Sent back over WebSocket after each chunk."""
    emotion:      str
    confidence:   float
    valence:      float
    arousal:      float
    all_probs:    Dict[str, float]
    text_weight:  Optional[float]
    audio_weight: Optional[float]
    chunk_index:  int = Field(..., description="Monotonically increasing chunk counter.")


class HealthResponse(BaseModel):
    status:       str
    checkpoint:   Optional[str]
    device:       str
    emotions:     list
    sample_rate:  int
    model_params: int