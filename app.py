# -*- coding: utf-8 -*-
"""NeuraType AI - Next Word Prediction & Text Completion App

Compatible with both direct Python execution, Gradio, and FastAPI backend.
"""

import os
import sys
import re
import pickle
import numpy as np

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Project settings
INPUT_LEN = 19
MODEL_PATH = os.path.join(BASE_DIR, "next_word_prediction.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# Lazy loading handles
_model = None
_tokenizer = None


def get_model_and_tokenizer():
    """Load model and tokenizer lazily to avoid heavy import overhead at startup."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    import tensorflow as tf

    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)

    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model, _tokenizer


def predict_next_word(text, top_k=5):
    """Predict next possible words given an input prefix sentence."""
    if not text or not text.strip():
        return "Please enter some text first."

    try:
        model, tokenizer = get_model_and_tokenizer()
    except Exception as e:
        return f"Error loading model: {e}"

    # Clean text to match training pipeline
    cleaned = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s']", "", cleaned)

    tokens = tokenizer.texts_to_sequences([cleaned])
    token_ids = tokens[0] if tokens else []

    if not token_ids:
        return "No recognized words found. Please enter another sentence."

    # Keep last INPUT_LEN tokens
    token_ids = token_ids[-INPUT_LEN:]

    # Pre-pad with zeros
    padded = np.zeros((1, INPUT_LEN), dtype=np.int32)
    padded[0, -len(token_ids) :] = token_ids

    # Predict probabilities
    predictions = model.predict(padded, verbose=0)[0]

    # Token IDs to exclude (padding 0 and OOV token)
    excluded_ids = {0}
    oov_id = tokenizer.word_index.get("<OOV>")
    if oov_id is not None:
        excluded_ids.add(oov_id)

    # Rank IDs descending
    ranked_ids = np.argsort(predictions)[::-1]

    results = []
    for token_id in ranked_ids:
        int_id = int(token_id)
        if int_id in excluded_ids:
            continue

        word = tokenizer.index_word.get(int_id)
        if not word:
            continue

        probability = float(predictions[int_id]) * 100
        results.append(f"{len(results) + 1}. **{word}** — {probability:.2f}%")

        if len(results) >= top_k:
            break

    if not results:
        return "No predictions available."

    return "\n\n".join(results)


def launch_web_server():
    """Launch the full-featured modern FastAPI + Web UI server."""
    import uvicorn
    from backend.main import app

    print("\n" + "=" * 60)
    print("🚀 Launching NeuraType AI Full Stack Server")
    print("🌐 Frontend UI & API available at: http://localhost:8000")
    print("📖 Interactive Swagger Docs at:    http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)


def launch_gradio():
    """Launch Gradio interface if installed."""
    try:
        import gradio as gr

        with gr.Blocks(title="NeuraType AI - Next Word Prediction") as demo:
            gr.Markdown("# 🧠 NeuraType AI - Next Word Prediction System")
            gr.Markdown(
                "Enter a sentence prefix and let the LSTM model predict the next possible words."
            )

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Enter your sentence",
                        placeholder="the future of artificial...",
                        lines=3,
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1, label="Top K Words"
                    )
                    predict_btn = gr.Button(
                        "Predict Next Word", variant="primary"
                    )

                with gr.Column():
                    output = gr.Markdown(label="Top Predictions")

            predict_btn.click(
                fn=predict_next_word, inputs=[input_text, top_k_slider], outputs=output
            )

        demo.launch(server_port=7860)
    except ImportError:
        print("Gradio is not installed. Launching the modern full-stack web UI instead...")
        launch_web_server()


if __name__ == "__main__":
    # If run with --gradio argument, run gradio; otherwise run the full stack server
    if "--gradio" in sys.argv:
        launch_gradio()
    else:
        launch_web_server()
