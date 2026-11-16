# 🧠 NeuraType AI — Next Word Prediction & Text Completion System

An end-to-end, deep learning powered Next Word Prediction and Autoregressive Text Completion web application. Built with an **LSTM (Long Short-Term Memory)** neural network trained in Keras/TensorFlow, an asynchronous high-performance **FastAPI** backend, and a glassmorphic **modern interactive web frontend**.

---

## 🌟 Key Features

### 🖥️ Frontend (Interactive Web Application)
- **Live Predictive Typing (Typeahead)**: As you type, the neural network computes the top likely next word in real time and displays it as a seamless **ghost suggestion** ahead of your cursor.
- **Keyboard Tab-to-Complete**: Press <kbd>Tab</kbd> or <kbd>→</kbd> to accept the predicted ghost word instantly.
- **One-Click Word Suggestion Chips**: Top candidate words are displayed with confidence badges (e.g. `intelligence 84.2%`). Clicking any chip appends the word and automatically chains into predicting the next word.
- **Probability Distribution Bar Chart**: Visualizes ranked confidence percentages for candidate words with animated gradient bars.
- **Multi-Word Auto-Complete Generator**: Iteratively generates sequences of up to 20 words with adjustable **Creativity / Temperature** (0.1 to 2.0) and selectable sampling strategies (**Greedy** vs **Creative Top-K Sampling**).
- **Neural Token Inspector**: Deep-dive diagnostic card showing:
  - Sanitized text
  - Extracted word tokens
  - Vocabulary integer IDs
  - 19-dimensional padded input vector sent into the LSTM embedding layer
  - Model inference latency in milliseconds
- **Preset Prompt Shelf**: One-click prompt examples across technology, conversational English, storytelling, and science.
- **Glassmorphic Dark / Light Theme**: Clean visual aesthetic with ambient glowing gradients, responsive layout, and theme persistence.

### ⚙️ Backend (FastAPI REST API)
- **High Performance & Asynchronous**: Built with FastAPI, Pydantic v2, and Uvicorn.
- **Interactive Swagger Docs**: Fully documented REST endpoints at `http://localhost:8000/docs`.
- **CORS Enabled**: Ready for decoupled frontend hosting or embedded integration.
- **Unified Single-Port Serving**: Mounts the static frontend at root (`/`) so the entire application runs effortlessly on a single port.
- **Temperature & Softmax Scaling**: Customizable temperature scaling on model logits for fine-grained control over prediction entropy.

---

## 🏗️ Architecture & Project Structure

```
c:\Users\HP\Desktop\next_word_prediction\
├── backend/
│   ├── __init__.py
│   ├── main.py                # FastAPI REST API & static file mount
│   ├── model.py               # ModelPredictor inference engine (loads Keras model & tokenizer)
│   └── schemas.py             # Pydantic data validation schemas
├── frontend/
│   ├── index.html             # Main web application UI
│   ├── css/
│   │   └── style.css          # Glassmorphic dark/light styling and animations
│   └── js/
│       └── app.js             # Client state, debounce typing, ghost text, charts
├── next_word_prediction.keras  # Trained Keras 3 LSTM neural network
├── tokenizer.pkl              # Pickled Keras Tokenizer (10,000 word vocabulary)
├── app.py                     # Dual-mode script (FastAPI server or Gradio UI)
├── run_server.py              # Launcher script that automatically opens your browser
├── start.bat                  # One-click Windows batch launcher
├── requirements.txt           # Python package dependencies
└── README.md                  # Complete documentation
```

---

## 🧠 Neural Network Model Specifications

| Parameter | Specification |
|---|---|
| **Architecture** | Sequential: `InputLayer(19)` ➔ `Embedding(10000, 128)` ➔ `LSTM(128)` ➔ `Dropout(0.2)` ➔ `Dense(10000, softmax)` |
| **Input Sequence Length** | 19 tokens (pre-padded with zeros) |
| **Vocabulary Size** | 10,000 words |
| **Framework** | Keras 3 / TensorFlow 2 |
| **Model Weights File** | `next_word_prediction.keras` (~32.4 MB) |
| **Tokenizer File** | `tokenizer.pkl` (~500 KB) |

---

## 🚀 Getting Started

### 1. Requirements & Prerequisites
- Python 3.10 or 3.11 installed.
- Windows, macOS, or Linux.

### 2. Install Dependencies
Open PowerShell or Command Prompt in this folder and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
You can start the server in any of the following ways:

#### Option A: One-Click Batch File (Windows)
Double-click:
```
start.bat
```

#### Option B: Python Launcher (Auto-opens browser)
```bash
python run_server.py
```

#### Option C: Direct FastAPI / Uvicorn Execution
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Then visit:
- **Web Application**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative Redoc API Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

#### Option D: Gradio Interface (Optional)
If you prefer the simple Gradio interface:
```bash
python app.py --gradio
```

---

## 📡 REST API Reference

### 1. Health Status
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 2. Model Specifications
- **Endpoint**: `GET /api/info`
- **Response**:
```json
{
  "status": "ready",
  "model_name": "Next Word LSTM",
  "model_type": "Keras 3 Sequential (Embedding -> LSTM -> Dropout -> Dense)",
  "input_sequence_length": 19,
  "vocab_size": 10000,
  "has_oov_token": false,
  "device": "CPU",
  "weights_size_mb": 32.45
}
```

### 3. Predict Next Words
- **Endpoint**: `POST /api/predict`
- **Request Body**:
```json
{
  "text": "the future of artificial",
  "top_k": 5,
  "temperature": 1.0
}
```
- **Response**:
```json
{
  "success": true,
  "input_text": "the future of artificial",
  "predictions": [
    {
      "rank": 1,
      "word": "intelligence",
      "probability": 0.8421,
      "percentage": 84.21
    },
    {
      "rank": 2,
      "word": "general",
      "probability": 0.0612,
      "percentage": 6.12
    }
  ],
  "top_word": "intelligence",
  "latency_ms": 18.5,
  "tokens_info": {
    "cleaned_text": "the future of artificial",
    "raw_tokens": ["the", "future", "of", "artificial"],
    "token_ids": [1, 423, 5, 891],
    "padded_sequence": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 423, 5, 891]
  }
}
```

### 4. Autoregressive Multi-Word Text Generation
- **Endpoint**: `POST /api/generate`
- **Request Body**:
```json
{
  "text": "i am looking forward to",
  "max_words": 5,
  "temperature": 0.8,
  "top_k": 5,
  "mode": "sample"
}
```
- **Response**:
```json
{
  "success": true,
  "original_text": "i am looking forward to",
  "generated_text": "hearing from you soon regarding",
  "completed_text": "i am looking forward to hearing from you soon regarding",
  "added_words": ["hearing", "from", "you", "soon", "regarding"],
  "latency_ms": 94.2
}
```

---

## 💡 How It Works Under The Hood
1. **Preprocessing**: The input string is normalized to lowercase and stripped of non-alphanumeric punctuation while preserving apostrophes.
2. **Tokenization**: Converted into sequence integers using the trained `tokenizer.pkl` mapping.
3. **Truncation & Zero-Padding**: The sequence is sliced to the last 19 tokens and pre-padded with zeroes to match the LSTM's required shape `(1, 19)`.
4. **Softmax Inference**: The model outputs logits across the 10,000 vocabulary tokens.
5. **Temperature Scaling**: Softmax temperature scales probability sharpness (lower temperature = more deterministic, higher = more diverse).
6. **Filtering**: Excludes padding index (0) and Out-Of-Vocabulary (`<OOV>`) token to return natural language words.
