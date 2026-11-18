"""Pydantic schemas for the Next Word Prediction API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateWord(BaseModel):
    """Candidate next word prediction item."""
    rank: int = Field(..., description="Rank among predictions (1 to K)")
    word: str = Field(..., description="Predicted next word")
    probability: float = Field(..., description="Probability score between 0.0 and 1.0")
    percentage: float = Field(..., description="Confidence percentage (0 to 100)")
    from_dataset: bool = Field(False, description="Whether this word directly matches the training dataset ground truth")
    dataset_match_order: Optional[int] = Field(None, description="N-gram match context length (e.g. 4 for 4-gram)")


class TokenDebugInfo(BaseModel):
    """Debug and diagnostic details about tokenization and sequence padding."""
    cleaned_text: str = Field(..., description="Sanitized input text")
    raw_tokens: List[str] = Field(default_factory=list, description="Extracted word tokens")
    token_ids: List[int] = Field(default_factory=list, description="Mapped tokenizer integer IDs")
    padded_sequence: List[int] = Field(default_factory=list, description="Fixed-length padded vector sent to LSTM")


class PredictRequest(BaseModel):
    """Request payload for next-word prediction."""
    text: str = Field(..., description="Input text / sentence prefix")
    top_k: int = Field(5, ge=1, le=25, description="Number of candidate words to return")
    temperature: float = Field(1.0, ge=0.1, le=3.0, description="Softmax temperature scaling factor")


class PredictResponse(BaseModel):
    """Response payload for next-word prediction."""
    success: bool = True
    input_text: str
    predictions: List[CandidateWord] = Field(default_factory=list)
    top_word: Optional[str] = None
    dataset_matched: bool = False
    latency_ms: float = 0.0
    tokens_info: Optional[TokenDebugInfo] = None
    error: Optional[str] = None


class GenerationStep(BaseModel):
    """Detailed metadata for a single step in text generation."""
    step: int
    chosen_word: str
    probability: float
    top_alternatives: List[CandidateWord] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    """Request payload for multi-word text generation."""
    text: str = Field(..., description="Starting seed sentence or phrase")
    max_words: int = Field(5, ge=1, le=50, description="Number of next words to generate sequentially")
    temperature: float = Field(0.8, ge=0.1, le=2.5, description="Sampling randomness temperature")
    top_k: int = Field(5, ge=1, le=20, description="Top-K candidate pool for sampling")
    mode: str = Field("greedy", description="Generation strategy: 'greedy' or 'sample'")


class GenerateResponse(BaseModel):
    """Response payload for multi-word text generation."""
    success: bool = True
    original_text: str
    generated_text: str
    completed_text: str
    added_words: List[str] = Field(default_factory=list)
    steps: List[GenerationStep] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Metadata about the loaded LSTM model and tokenizer."""
    status: str
    model_name: str
    model_type: str
    input_sequence_length: int
    vocab_size: int
    has_oov_token: bool
    device: str
    weights_size_mb: float


class HealthResponse(BaseModel):
    """Service health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class ExamplePrompt(BaseModel):
    """Sample prompt for quick testing."""
    category: str
    prompt: str
    description: str
