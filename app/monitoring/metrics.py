import time
import logging
import threading
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

PREDICTIONS_COUNTER = Counter(
    "sentiment_predictions_total",
    "Total number of sentiment predictions",
    ["sentiment"],
)

PREDICTION_LATENCY = Histogram(
    "sentiment_prediction_latency_seconds",
    "Sentiment prediction latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

TEXT_LENGTH_HISTOGRAM = Histogram(
    "sentiment_text_length",
    "Length of analyzed text",
    buckets=(10, 20, 50, 100, 200, 500, 1000),
)

CONFIDENCE_HISTOGRAM = Histogram(
    "sentiment_confidence",
    "Confidence of sentiment predictions",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
)

metrics_server = None


def start_metrics_server(port: int = 8000) -> None:
    """
    Start Prometheus metrics server.

    Args:
        port: Port to expose metrics on
    """

    global metrics_server
    if metrics_server is None:
        try:
            start_http_server(port)
            logger.info(f"Started metrics server on port {port}")
            metrics_server = True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")


def record_prediction_metrics(
    text_length: int,
    sentiment: str,
    confidence: float,
    prediction_time: Optional[float] = None,
) -> None:
    """
    Record metrics for a sentiment prediction.

    Args:
        text_length: Length of the analyzed text
        sentiment: Predicted sentiment
        confidence: Confidence score of prediction
        prediction_time: Time taken for prediction (seconds)
    """

    try:
        PREDICTIONS_COUNTER.labels(sentiment=sentiment).inc()

        TEXT_LENGTH_HISTOGRAM.observe(text_length)

        CONFIDENCE_HISTOGRAM.observe(confidence)

        if prediction_time is not None:
            PREDICTION_LATENCY.observe(prediction_time)
    except Exception as e:
        logger.error(f"Error recording metrics: {str(e)}")
