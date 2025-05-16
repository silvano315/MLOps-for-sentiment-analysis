from app.models.prediction import predict_sentiment
from data.datasets.download_datasets import download_dataset


def test_tweet_eval_download_and_preprocess():
    """Test downloading and preprocessing the TweetEval dataset."""

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = download_dataset("tweet_eval", "sentiment", tmp_dir)

        assert "dataset" in result
        assert "info" in result
        assert result["info"]["name"] == "tweet_eval"

        dataset = result["dataset"]
        if "train" in dataset:
            sample = dataset["train"][0]
            assert "text" in sample
            assert "label" in sample


def test_model_prediction_on_tweet_eval():
    """Test model prediction on TweetEval samples."""

    import tempfile

    from data.datasets.download_datasets import download_and_prepare_datasets

    with tempfile.TemporaryDirectory() as tmp_dir:
        datasets = download_and_prepare_datasets(["tweet_eval"], tmp_dir)

        assert "tweet_eval" in datasets

        dataset = datasets["tweet_eval"]["dataset"]
        samples = dataset["test"].select(range(5))
        texts = samples["text"]
        predictions = predict_sentiment(texts)

        assert len(predictions) == len(texts)
        for pred in predictions:
            assert "sentiment" in pred
            assert "confidence" in pred
            assert pred["sentiment"] in ["negative", "neutral", "positive"]
