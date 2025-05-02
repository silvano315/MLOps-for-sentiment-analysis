import pytest

from app.models.model_loader import SentimentModel
from app.models.prediction import predict_sentiment

def test_model_initialization():
    """Test that the model can be initialized."""
    model = SentimentModel()
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None

def test_sentiment_prediction():
    """Test basic sentiment prediction functionality."""
    texts = [
        "I love this product!",
        "This is absolutely terrible.",
        "It's okay, nothing special."
    ]

    results = predict_sentiment(texts)

    assert len(results) == 3
    for result in results:
        assert "text" in result
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["sentiment"] in ["negative", "neutral", "positive"]
        assert 0 <= result["confidence"] <= 1
    
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"