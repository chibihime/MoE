"""api/routes/text.py — POST /predict/text"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.schemas import EmotionResponse
from api.dependencies import get_pipeline
from inference.pipeline import EmotionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])


class TextRequest(BaseModel):
    text: str


@router.post("/text", response_model=EmotionResponse)
async def predict_text(
    body: TextRequest,
    pipe: EmotionPipeline = Depends(get_pipeline),
):
    """
    Predict emotion from Japanese text.
    """
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text field is empty.")

    try:
        result = pipe.predict(text=body.text.strip())
        return EmotionResponse(**result.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Text prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")