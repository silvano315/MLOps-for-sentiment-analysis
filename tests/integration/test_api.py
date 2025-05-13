import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "system" in data


def test_readiness_endpoint():
    """Test the readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_liveness_endpoint():
    """Test the liveness check endpoint."""
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_sentiment_analysis_endpoint():
    """Test the sentiment analysis endpoint."""
    response = client.post("/api/v1/sentiment", json={"text": "I love this product!"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "I love this product!"
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert 0 <= data["confidence"] <= 1
    assert "probabilities" in data


def test_batch_sentiment_analysis_endpoint():
    """Test the batch sentiment analysis endpoint."""
    texts = [
        "I love this product!",
        "This is absolutely terrible.",
        "It's okay, nothing special.",
    ]

    response = client.post("/api/v1/sentiment/batch", json={"texts": texts})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3

    for i, result in enumerate(data["results"]):
        assert result["text"] == texts[i]
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result


def test_empty_batch_request():
    """Test error handling for empty batch requests."""
    response = client.post("/api/v1/sentiment/batch", json={"texts": []})
    assert response.status_code == 400


def test_invalid_text_request():
    """Test error handling for invalid text requests."""
    response = client.post("/api/v1/sentiment", json={"text": ""})
    assert response.status_code == 422
