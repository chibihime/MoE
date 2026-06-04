"""api/routes/health.py — GET /health"""

from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api.dependencies import get_pipeline
from inference.pipeline import EmotionPipeline

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(pipe: EmotionPipeline = Depends(get_pipeline)):
    """
    Returns pipeline metadata and confirms the server is ready.
    Use this as your load-balancer health check endpoint.
    """
    info = pipe.info
    return HealthResponse(
        status="ok",
        checkpoint=info["checkpoint"],
        device=info["device"],
        emotions=info["emotions"],
        sample_rate=info["sample_rate"],
        model_params=info["model_params"],
    )