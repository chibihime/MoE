"""api/routes/text.py — POST /predict/text"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import TextRequest, EmotionResponse
from api.dependencies import get_pipeline
from inference.pipeline import EmotionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])


@router.post("/text", response_model=EmotionResponse)
async def predict_text(
    body: TextRequest,
    pipe: EmotionPipeline = Depends(get_pipeline),
):
    """
    Predict emotion from a Japanese text string.

    Returns the predicted emotion label, confidence, valence/arousal scores,
    and the full probability distribution over all 7 classes.
    """
    try:
        result = pipe.predict_text(body.text)
        return EmotionResponse(**result.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Text prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")