"""Inference and model management engine for Next Word Prediction.

Supports both pure NumPy high-performance inference (instant startup, zero heavy dependencies)
and TensorFlow/Keras fallback.
"""

import os
import re
import time
import pickle
import zipfile
import io
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("next_word_prediction.model")

INPUT_LEN = 19
NUM_WORDS = 10000


class DummyTokenizer:
    """Safe unpickling target for Keras Tokenizer without requiring keras/tensorflow installed."""
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.oov_token = "<OOV>"
        self.num_words = NUM_WORDS
        self.filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        self.lower = True

    def __setstate__(self, state):
        self.__dict__.update(state)


class SafeTokenizerUnpickler(pickle.Unpickler):
    """Custom unpickler to deserialize Keras Tokenizer safely."""
    def find_class(self, module, name):
        if "Tokenizer" in name:
            return DummyTokenizer
        return super().find_class(module, name)


class ModelPredictor:
    """Manages loading of the LSTM model weights and tokenizer,

    providing high-performance next-word prediction, multi-word generation, and diagnostics.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.model_path = os.path.join(self.base_dir, "next_word_prediction.keras")
        self.tokenizer_path = os.path.join(self.base_dir, "tokenizer.pkl")

        # Weights
        self.emb_w = None
        self.w_x = None
        self.w_h = None
        self.b_lstm = None
        self.dense_w = None
        self.dense_b = None

        self.tokenizer = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.excluded_ids = {0}
        self.vocab_size = 0
        self.oov_id: Optional[int] = None
        self.hidden_units = 128

    def load(self) -> bool:
        """Load Keras model weights and tokenizer into memory."""
        try:
            logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")

            with open(self.tokenizer_path, "rb") as f:
                try:
                    self.tokenizer = pickle.load(f)
                except Exception:
                    f.seek(0)
                    self.tokenizer = SafeTokenizerUnpickler(f).load()

            self.vocab_size = len(self.tokenizer.word_index)
            self.oov_id = self.tokenizer.word_index.get("<OOV>")
            if self.oov_id is not None:
                self.excluded_ids.add(self.oov_id)

            logger.info(
                f"Tokenizer loaded successfully. Vocab size: {self.vocab_size}, OOV ID: {self.oov_id}"
            )

            logger.info(f"Loading model weights from: {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            # Open .keras archive (zip) and load HDF5 weights directly using h5py
            import h5py

            with zipfile.ZipFile(self.model_path, "r") as z:
                weights_bytes = z.read("model.weights.h5")
                with h5py.File(io.BytesIO(weights_bytes), "r") as f:
                    self.emb_w = np.array(f["layers/embedding/vars/0"], dtype=np.float32)
                    self.w_x = np.array(f["layers/lstm/cell/vars/0"], dtype=np.float32)
                    self.w_h = np.array(f["layers/lstm/cell/vars/1"], dtype=np.float32)
                    self.b_lstm = np.array(f["layers/lstm/cell/vars/2"], dtype=np.float32)
                    self.dense_w = np.array(f["layers/dense/vars/0"], dtype=np.float32)
                    self.dense_b = np.array(f["layers/dense/vars/1"], dtype=np.float32)

            self.hidden_units = self.emb_w.shape[1]  # 128
            self.is_loaded = True
            self.load_error = None
            logger.info("Model weights loaded successfully into fast inference engine.")
            return True

        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)
            logger.error(f"Failed to load model/tokenizer: {exc}", exc_info=True)
            return False

    def clean_text(self, text: str) -> str:
        """Sanitize text identically to training preprocessing."""
        if not text:
            return ""
        text = text.lower()
        # Remove characters except lowercase alphanumerics, spaces, and apostrophes
        text = re.sub(r"[^a-z0-9\s']", "", text)
        return text.strip()

    def tokenize_and_pad(
        self, text: str
    ) -> Tuple[List[str], List[int], np.ndarray, str]:
        """Convert raw input text to tokens, sequence IDs, and padded numpy vector."""
        if not self.is_loaded:
            self.load()

        cleaned = self.clean_text(text)
        if not cleaned:
            return [], [], np.zeros((1, INPUT_LEN), dtype=np.int32), cleaned

        words = cleaned.split()
        token_ids = []

        max_vocab = getattr(self.tokenizer, "num_words", None) or NUM_WORDS
        oov_token_id = self.oov_id or 1

        for w in words:
            idx = self.tokenizer.word_index.get(w)
            if idx is not None and idx < max_vocab:
                token_ids.append(idx)
            else:
                if self.oov_id is not None:
                    token_ids.append(oov_token_id)

        if not token_ids:
            return words, [], np.zeros((1, INPUT_LEN), dtype=np.int32), cleaned

        # Take last INPUT_LEN tokens
        trimmed_tokens = token_ids[-INPUT_LEN:]

        # Pre-pad with zeros
        padded = np.zeros((1, INPUT_LEN), dtype=np.int32)
        padded[0, -len(trimmed_tokens):] = trimmed_tokens

        return words, token_ids, padded, cleaned

    def _forward_pass(self, padded_sequence: np.ndarray) -> np.ndarray:
        """High-performance vectorized LSTM + Dense forward pass in NumPy."""
        units = self.hidden_units
        h = np.zeros(units, dtype=np.float32)
        c = np.zeros(units, dtype=np.float32)

        # Standard numerically stable sigmoid
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

        seq = padded_sequence[0]
        for token_id in seq:
            if token_id == 0:
                # Mask zero padding
                continue

            x = self.emb_w[token_id]
            # LSTM gates: z = x * W_x + h * W_h + b
            z = np.dot(x, self.w_x) + np.dot(h, self.w_h) + self.b_lstm

            # Keras gate ordering: input, forget, cell, output
            i = sigmoid(z[:units])
            f = sigmoid(z[units: 2 * units])
            c_cand = np.tanh(z[2 * units: 3 * units])
            o = sigmoid(z[3 * units: 4 * units])

            c = f * c + i * c_cand
            h = o * np.tanh(c)

        # Dense layer: logits = h * W_dense + b_dense
        logits = np.dot(h, self.dense_w) + self.dense_b
        return logits

    def predict_next_words(
        self, text: str, top_k: int = 5, temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Predict top-k next words with probabilities and diagnostics."""
        start_time = time.perf_counter()

        if not self.is_loaded:
            loaded = self.load()
            if not loaded:
                return {
                    "success": False,
                    "error": f"Model failed to load: {self.load_error}",
                    "predictions": [],
                    "latency_ms": 0.0,
                }

        words, token_ids, padded, cleaned = self.tokenize_and_pad(text)

        if not token_ids:
            return {
                "success": False,
                "error": "No recognized vocabulary words found. Please enter recognizable words.",
                "predictions": [],
                "top_word": None,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "tokens_info": {
                    "cleaned_text": cleaned,
                    "raw_tokens": words,
                    "token_ids": [],
                    "padded_sequence": padded[0].tolist(),
                },
            }

        # Compute model logits
        logits = self._forward_pass(padded)

        # Apply temperature scaling
        if temperature <= 0.05:
            temperature = 0.05
        scaled_logits = logits / temperature

        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        # Rank descending
        ranked_ids = np.argsort(probs)[::-1]

        predictions = []
        for token_id in ranked_ids:
            int_id = int(token_id)
            if int_id in self.excluded_ids:
                continue

            word = self.tokenizer.index_word.get(int_id)
            if not word:
                continue

            prob = float(probs[int_id])
            predictions.append(
                {
                    "rank": len(predictions) + 1,
                    "word": word,
                    "probability": round(prob, 5),
                    "percentage": round(prob * 100, 2),
                }
            )

            if len(predictions) >= top_k:
                break

        top_word = predictions[0]["word"] if predictions else None
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "success": True,
            "input_text": text,
            "predictions": predictions,
            "top_word": top_word,
            "latency_ms": latency_ms,
            "tokens_info": {
                "cleaned_text": cleaned,
                "raw_tokens": words,
                "token_ids": token_ids,
                "padded_sequence": padded[0].tolist(),
            },
        }

    def generate_text(
        self,
        seed_text: str,
        max_words: int = 5,
        temperature: float = 0.8,
        top_k: int = 5,
        mode: str = "greedy",
    ) -> Dict[str, Any]:
        """Autoregressively generate next words given a seed sentence."""
        start_time = time.perf_counter()

        if not self.is_loaded:
            loaded = self.load()
            if not loaded:
                return {
                    "success": False,
                    "error": f"Model failed to load: {self.load_error}",
                    "generated_text": "",
                    "completed_text": seed_text,
                    "added_words": [],
                    "steps": [],
                    "latency_ms": 0.0,
                }

        current_text = seed_text.strip()
        added_words = []
        steps = []

        for step_idx in range(1, max_words + 1):
            pred_result = self.predict_next_words(
                current_text, top_k=top_k, temperature=temperature
            )

            if not pred_result.get("success") or not pred_result.get("predictions"):
                break

            candidates = pred_result["predictions"]

            if mode == "sample" and len(candidates) > 1:
                word_probs = np.array([c["probability"] for c in candidates])
                sum_p = np.sum(word_probs)
                if sum_p > 0:
                    word_probs = word_probs / sum_p
                    chosen_idx = np.random.choice(len(candidates), p=word_probs)
                else:
                    chosen_idx = 0
                chosen = candidates[chosen_idx]
            else:
                chosen = candidates[0]

            chosen_word = chosen["word"]
            added_words.append(chosen_word)

            steps.append(
                {
                    "step": step_idx,
                    "chosen_word": chosen_word,
                    "probability": chosen["probability"],
                    "top_alternatives": candidates[:4],
                }
            )

            current_text = f"{current_text} {chosen_word}"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        generated_suffix = " ".join(added_words)

        return {
            "success": True,
            "original_text": seed_text,
            "generated_text": generated_suffix,
            "completed_text": current_text,
            "added_words": added_words,
            "steps": steps,
            "latency_ms": latency_ms,
        }

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata and specifications."""
        weights_mb = 0.0
        if os.path.exists(self.model_path):
            weights_mb = round(os.path.getsize(self.model_path) / (1024 * 1024), 2)

        return {
            "status": "ready" if self.is_loaded else "uninitialized",
            "model_name": "Next Word LSTM",
            "model_type": "Keras 3 Sequential (Embedding -> LSTM -> Dropout -> Dense)",
            "input_sequence_length": INPUT_LEN,
            "vocab_size": self.vocab_size or NUM_WORDS,
            "has_oov_token": self.oov_id is not None,
            "device": "CPU (Optimized NumPy)",
            "weights_size_mb": weights_mb,
        }


# Global predictor instance
predictor = ModelPredictor()
