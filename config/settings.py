import os
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    "Application settings"

    model_config = SettingsConfigDict(env_file=".venv", env_file_encoding="utf-8")

    # API configuration
    API_TITLE: str = "Sentiment Analysis API"
    API_DESCRIPTION: str = "API for sentiment analysis of social media texts"
    API_VERSION: str = "0.1.0"

    # Model configuration
    MODEL_NAME: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    MODEL_CACHE_DIR: Optional[str] = None

    # Data configuration
    PRIMARY_DATASET: str = "tweet_eval"
    PRIMARY_DATASET_CONFIG: str = "sentiment"
    SECONDARY_DATASET: str = "mteb/amazon_reviews_multi"
    SECONDARY_DATASET_CONFIG: str = "en"

    # Monitoring configuration
    METRICS_PORT: int = 8000

    # RapidAPI configuration
    RAPIDAPI_KEY: Optional[str] = None
    TWITTER_API_HOST: str = "twitter154.p.rapidapi.com"


settings = Settings()
