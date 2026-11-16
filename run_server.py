"""Convenient launcher script for NeuraType AI.

Starts the FastAPI server and automatically opens the browser.
"""

import os
import sys
import time
import webbrowser
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def open_browser():
    """Wait briefly for server to bind then open the browser."""
    time.sleep(1.8)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🧠 Starting NeuraType AI - Next Word Prediction & Text Completion")
    print("=" * 65)
    print("  • Web Application:  http://127.0.0.1:8000")
    print("  • API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 65 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
