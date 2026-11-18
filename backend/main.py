"""FastAPI main application entrypoint for Next Word Prediction."""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add current directory to path if needed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.schemas import (
    PredictRequest,
    PredictResponse,
    GenerateRequest,
    GenerateResponse,
    ModelInfoResponse,
    HealthResponse,
    ExamplePrompt,
)
from backend.model import predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("next_word_prediction.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event: load ML model and tokenizer on server startup."""
    logger.info("Initializing Next Word Prediction engine on startup...")
    try:
        predictor.load()
    except Exception as e:
        logger.error(f"Error during startup model load: {e}")
    yield
    logger.info("Shutting down Next Word Prediction engine.")


app = FastAPI(
    title="NeuraType AI - Next Word Prediction API",
    description="Deep learning LSTM API for real-time next-word prediction and multi-word sentence completion.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for seamless local and remote browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIRouter allows endpoints to match both with and without '/api' prefix (crucial for Vercel)
router = APIRouter()


@router.get("/download", tags=["System"])
def download_project_zip():
    """Download the complete project packaged as a ZIP archive."""
    for base in [PROJECT_ROOT, os.getcwd(), os.path.dirname(__file__)]:
        zip_candidate = os.path.abspath(os.path.join(base, "next_word_prediction.zip"))
        if os.path.exists(zip_candidate):
            return FileResponse(zip_candidate, filename="next_word_prediction.zip", media_type="application/zip")
        parent_candidate = os.path.abspath(os.path.join(base, "..", "next_word_prediction.zip"))
        if os.path.exists(parent_candidate):
            return FileResponse(parent_candidate, filename="next_word_prediction.zip", media_type="application/zip")

    # If not found, create it on the fly
    zip_path = os.path.abspath(os.path.join(PROJECT_ROOT, "next_word_prediction.zip"))
    import shutil
    shutil.make_archive(os.path.join(PROJECT_ROOT, "next_word_prediction"), "zip", PROJECT_ROOT)
    return FileResponse(zip_path, filename="next_word_prediction.zip", media_type="application/zip")


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Verify service and model health status."""
    if not predictor.is_loaded:
        predictor.load()
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "ready_to_load",
        model_loaded=predictor.is_loaded,
        version="1.0.0",
    )


@router.get("/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """Retrieve specifications and architecture info about the loaded model."""
    if not predictor.is_loaded:
        predictor.load()
    info = predictor.get_info()
    return ModelInfoResponse(**info)


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: PredictRequest):
    """Predict the top-k next words for the provided input sentence prefix."""
    text = payload.text.strip()
    if not text:
        return PredictResponse(
            success=False,
            input_text=payload.text,
            predictions=[],
            top_word=None,
            latency_ms=0.0,
            error="Please provide input text.",
        )

    result = predictor.predict_next_words(
        text=text,
        top_k=payload.top_k,
        temperature=payload.temperature,
    )
    return PredictResponse(**result)


@router.post("/generate", response_model=GenerateResponse, tags=["Generation"])
def generate(payload: GenerateRequest):
    """Sequentially generate next words starting from a seed phrase."""
    text = payload.text.strip()
    if not text:
        return GenerateResponse(
            success=False,
            original_text=payload.text,
            generated_text="",
            completed_text="",
            added_words=[],
            steps=[],
            latency_ms=0.0,
            error="Please provide starting seed text.",
        )

    result = predictor.generate_text(
        seed_text=text,
        max_words=payload.max_words,
        temperature=payload.temperature,
        top_k=payload.top_k,
        mode=payload.mode,
    )
    return GenerateResponse(**result)


@router.get("/examples", response_model=List[ExamplePrompt], tags=["Examples"])
def get_examples():
    """Return preconfigured interesting prompts from the Shakespeare dataset."""
    return [
        ExamplePrompt(
            category="Dataset Quote",
            prompt="before we proceed any further hear me",
            description="Predicts exact ground-truth next word: 'speak'.",
        ),
        ExamplePrompt(
            category="Dataset Quote",
            prompt="you are all resolved rather to die than to",
            description="Predicts exact ground-truth next word: 'famish'.",
        ),
        ExamplePrompt(
            category="Dataset Quote",
            prompt="first you know caius marcius is chief",
            description="Predicts exact ground-truth next word: 'enemy'.",
        ),
        ExamplePrompt(
            category="Famous Quote",
            prompt="to be or not to",
            description="Predicts famous continuation: 'be'.",
        ),
        ExamplePrompt(
            category="Dataset Quote",
            prompt="what light through yonder window",
            description="Predicts exact ground-truth next word: 'breaks'.",
        ),
    ]


# Include router under both '/api' and root
app.include_router(router, prefix="/api")
app.include_router(router)


def find_index_html() -> str:
    """Find index.html across multiple possible static directories."""
    candidates = [
        os.path.join(PROJECT_ROOT, "public", "index.html"),
        os.path.join(PROJECT_ROOT, "frontend", "index.html"),
        os.path.join(os.getcwd(), "public", "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
        "/var/task/public/index.html",
        "/var/task/frontend/index.html",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


@app.get("/")
def serve_root():
    """Serve index.html at root if static files are routed to FastAPI."""
    idx = find_index_html()
    if idx:
        return FileResponse(idx)
    return {"message": "NeuraType AI Backend API is online.", "docs": "/docs"}


# Mount static assets if directories exist locally
for d_name in ["public", "frontend"]:
    static_dir = os.path.join(PROJECT_ROOT, d_name)
    if os.path.exists(static_dir):
        css_dir = os.path.join(static_dir, "css")
        js_dir = os.path.join(static_dir, "js")
        if os.path.exists(css_dir):
            app.mount("/css", StaticFiles(directory=css_dir), name=f"css_{d_name}")
        if os.path.exists(js_dir):
            app.mount("/js", StaticFiles(directory=js_dir), name=f"js_{d_name}")
        app.mount("/static", StaticFiles(directory=static_dir, html=True), name=f"static_{d_name}")
        break


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
