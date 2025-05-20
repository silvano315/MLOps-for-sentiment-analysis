import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "tweet_eval": {
        "name": "tweet_eval",
        "config": "sentiment",
        "label_mapping": {0: "negative", 1: "neutral", 2: "positive"},
        "text_column": "text",
        "label_column": "label",
    },
    "mteb/amazon_reviews_multi": {
        "name": "mteb/amazon_reviews_multi",
        "config": "en",
        "label_mapping": {
            0: "negative",  # Stars 1-2 -> negative
            1: "neutral",  # Stars 3 -> neutral
            2: "positive",  # Stars 4-5 -> positive
        },
        "text_column": "text",
        "label_column": "label",
        "preprocessing": "map_amazon_stars",
    },
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset configuration dictionary
    """

    if dataset_name not in DATASET_REGISTRY:
        logger.warning(f"Dataset {dataset_name} not found in registry")
        return {}

    return DATASET_REGISTRY[dataset_name]


def map_amazon_stars(stars: int) -> int:
    """
    Map Amazon review stars to sentiment classes.

    Args:
        stars: Amazon review stars (0-4)

    Returns:
        Sentiment class (0=negative, 1=neutral, 2=positive)
    """

    if stars <= 1:
        return 0  # negative
    elif stars == 2:
        return 1  # neutral
    else:
        return 2  # positive
