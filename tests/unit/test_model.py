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
        "It's okay, nothing special.",
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
    assert results[2]["sentiment"] == "neutral"


def test_confidence_matches_probs():
    """Test to ensure confidence matches the max value in probabilities."""

    texts = ["I love it!"]
    result = predict_sentiment(texts)[0]
    max_prob = max(result["probabilities"].values())
    assert abs(result["confidence"] - max_prob) <= 1e-6
