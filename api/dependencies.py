import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
api/dependencies.py

Pipeline singleton — loaded once at startup, injected into every route.

FastAPI's Depends() mechanism means each route handler receives the same
loaded pipeline without re-loading the model per request.
"""

import os
from functools import lru_cache

from fastapi import Depends
from inference.pipeline import EmotionPipeline

_pipeline: EmotionPipeline | None = None


def init_pipeline() -> EmotionPipeline:
    """
    Load the pipeline at server startup.
    Called once from main.py lifespan context.
    """
    global _pipeline

    checkpoint = os.getenv("CHECKPOINT_PATH", "outputs/checkpoints/best.pt")
    config_dir = os.getenv("CONFIG_DIR",      "configs/")
    device     = os.getenv("DEVICE",          None)     # auto-detect if unset

    _pipeline = EmotionPipeline.load(
        checkpoint=checkpoint,
        config_dir=config_dir,
        device=device,
    )
    _pipeline.warmup()
    return _pipeline


def get_pipeline() -> EmotionPipeline:
    """
    FastAPI dependency — inject the loaded pipeline into route handlers.

    Usage in a route:
        from api.dependencies import get_pipeline
        from inference.pipeline import EmotionPipeline

        @router.post("/predict/text")
        async def predict_text(
            body: TextRequest,
            pipe: EmotionPipeline = Depends(get_pipeline),
        ):
            ...
    """
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialised. "
            "Ensure init_pipeline() is called at startup."
        )
    return _pipeline