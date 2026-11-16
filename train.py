"""Model training script for Next Word Prediction LSTM.

Trains an LSTM language model on a text corpus for N epochs (default: 10)
and saves next_word_prediction.keras and tokenizer.pkl.
"""

import os
import sys
import re
import pickle
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default training sample corpus if no external file provided
DEFAULT_CORPUS = """
Artificial intelligence and machine learning are transforming the future of technology and human life.
Deep learning models such as recurrent neural networks and long short-term memory networks are designed to process sequential data and natural language.
In the modern world, computer algorithms analyze massive amounts of text to predict the next word in a sentence.
Once upon a time in a distant kingdom, the king and the duke ruled over the castle and the land.
Every morning the citizens would gather in the royal courtyard to discuss the future of the kingdom and the laws of the realm.
Science and technology continue to push the boundaries of human knowledge and exploration across the universe.
Computer vision, natural language processing, and neural language modeling are foundational pillars of modern artificial intelligence.
I am looking forward to exploring new frontiers and discovering innovative solutions to complex problems.
The quick brown fox jumps over the lazy dog in the middle of the quiet forest.
Language models learn probability distributions over sequences of words to generate coherent and meaningful sentences.
"""

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def prepare_data(text: str, max_sequence_len: int = 20, vocab_size: int = 10000):
    """Tokenize text and generate sequence windows of length max_sequence_len."""
    cleaned = clean_text(text)
    words = cleaned.split()
    print(f"Total words in corpus: {len(words)}")

    # Build vocabulary mapping
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    word_index = {"<OOV>": 1}
    for idx, (w, _) in enumerate(sorted_words[:vocab_size - 2], start=2):
        word_index[w] = idx

    index_word = {v: k for k, v in word_index.items()}

    # Convert words to integer token IDs
    tokens = [word_index.get(w, 1) for w in words]

    # Create sliding sequence windows
    sequences = []
    for i in range(len(tokens) - max_sequence_len):
        sequences.append(tokens[i : i + max_sequence_len])

    sequences = np.array(sequences, dtype=np.int32)
    X = sequences[:, :-1]  # 19 tokens input
    y = sequences[:, -1]   # 1 token target

    return X, y, word_index, index_word

def train_model(epochs: int = 10, batch_size: int = 32, corpus_file: str = None):
    print("=" * 65)
    print(f"🧠 Training Next Word Prediction Model for {epochs} Epochs")
    print("=" * 65)

    if corpus_file and os.path.exists(corpus_file):
        print(f"Loading custom training data from: {corpus_file}")
        with open(corpus_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Using standard built-in training corpus...")
        text = DEFAULT_CORPUS * 50  # Repeat to build rich sequence dataset

    X, y, word_index, index_word = prepare_data(text, max_sequence_len=20, vocab_size=10000)
    print(f"Training sequences generated: {len(X)}")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")

    # Save tokenizer pickle
    tokenizer_obj = type("Tokenizer", (), {
        "word_index": word_index,
        "index_word": index_word,
        "oov_token": "<OOV>",
        "num_words": 10000,
        "filters": '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        "lower": True,
        "texts_to_sequences": lambda self, texts: [[self.word_index.get(w, 1) for w in clean_text(texts[0]).split()]]
    })()

    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer_obj, f)
    print(f"Saved tokenizer to: {tokenizer_path}")

    # Try Keras / TensorFlow for training
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

        print(f"TensorFlow {tf.__version__} detected. Building neural network architecture...")
        model = Sequential([
            Embedding(input_dim=10000, output_dim=128, mask_zero=True),
            LSTM(128),
            Dropout(0.2),
            Dense(10000, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.summary()

        print(f"\nStarting {epochs} epochs of training...\n")
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        model_path = os.path.join(PROJECT_ROOT, "next_word_prediction.keras")
        model.save(model_path)
        print(f"\n🎉 Model successfully trained and saved to: {model_path}")
        return True

    except ImportError:
        print("TensorFlow not installed locally for training. Run 'pip install tensorflow' to train natively.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Next Word Prediction LSTM")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--data", type=str, default=None, help="Path to text corpus file")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, corpus_file=args.data)
