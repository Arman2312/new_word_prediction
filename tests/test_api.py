"""Integration tests for Next Word Prediction."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.main import app

def test_api():
    with TestClient(app) as client:
        # 1. Health
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["model_loaded"] is True

        # 2. Model Info
        res = client.get("/api/info")
        assert res.status_code == 200
        assert res.json()["status"] == "ready"

        # 3. Predict
        res = client.post("/api/predict", json={"text": "the future of", "top_k": 5})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["predictions"]) > 0
        assert data["tokens_info"]["cleaned_text"] == "the future of"

        # 4. Generate
        res = client.post("/api/generate", json={"text": "once upon a", "max_words": 3})
        assert res.status_code == 200
        assert res.json()["success"] is True
        assert len(res.json()["added_words"]) == 3

        # 5. Frontend static index
        res = client.get("/")
        assert res.status_code == 200
        assert "NeuraType" in res.text

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_api()
