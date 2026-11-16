"""FastAPI main application entrypoint for Next Word Prediction."""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
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
        success = predictor.load()
        if success:
            logger.info("Predictor loaded and ready for queries.")
        else:
            logger.warning(
                f"Predictor failed startup load: {predictor.load_error}. Will retry on request."
            )
    except Exception as e:
        logger.error(f"Error during model startup load: {e}")
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


# API Endpoints
@app.get("/api/download", tags=["System"])
def download_project_zip():
    """Download the complete project packaged as a ZIP archive."""
    zip_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "next_word_prediction.zip"))
    if not os.path.exists(zip_path):
        import shutil
        shutil.make_archive(os.path.join(PROJECT_ROOT, "..", "next_word_prediction"), "zip", PROJECT_ROOT)
    return FileResponse(zip_path, filename="next_word_prediction.zip", media_type="application/zip")


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Verify service and model health status."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "ready_to_load",
        model_loaded=predictor.is_loaded,
        version="1.0.0",
    )


@app.get("/api/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """Retrieve specifications and architecture info about the loaded model."""
    info = predictor.get_info()
    return ModelInfoResponse(**info)


@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
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


@app.post("/api/generate", response_model=GenerateResponse, tags=["Generation"])
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


@app.get("/api/examples", response_model=List[ExamplePrompt], tags=["Examples"])
def get_examples():
    """Return preconfigured interesting prompts for testing."""
    return [
        ExamplePrompt(
            category="Technology",
            prompt="the future of artificial",
            description="Predicts domain technology concepts like 'intelligence'.",
        ),
        ExamplePrompt(
            category="Daily Conversation",
            prompt="i am looking forward to",
            description="Common conversational English phrasing.",
        ),
        ExamplePrompt(
            category="Narrative",
            prompt="once upon a time in a",
            description="Storytelling and narrative sequences.",
        ),
        ExamplePrompt(
            category="Science & Research",
            prompt="machine learning algorithms are designed to",
            description="Technical phrase completion.",
        ),
        ExamplePrompt(
            category="Inquiry",
            prompt="what is the most important",
            description="Question formulation prediction.",
        ),
    ]


# Mount frontend static files directory
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    logger.warning(
        f"Frontend directory not found at {FRONTEND_DIR}. Only API endpoints will be served."
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
