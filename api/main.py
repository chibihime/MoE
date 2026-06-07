import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
api/main.py

FastAPI application for MoE Emotion Predictor.

Endpoints:
    GET  /health              — pipeline info + status
    POST /predict/text        — predict from Japanese text
    POST /predict/file        — predict from uploaded audio file
    WS   /predict/stream      — real-time prediction from PCM stream

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

The pipeline is loaded once at startup and shared across all requests.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import text, audio_file, stream, health, sample
from api.dependencies import init_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline once at startup, clean up on shutdown."""
    logger.info("Loading EmotionPipeline...")
    init_pipeline()
    logger.info("Pipeline ready. Server accepting requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="MoE Emotion Predictor",
    description=(
        "Real-time Japanese emotion prediction from text and audio. "
        "Returns one of 7 discrete emotion classes plus continuous "
        "valence and arousal scores."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chibihime.github.io"],      
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(text.router,       prefix="/predict")
app.include_router(audio_file.router, prefix="/predict")
app.include_router(stream.router,     prefix="/predict")
app.include_router(sample.router,     prefix="/predict")