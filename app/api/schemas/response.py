from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Dict, List

class SentimentResponse(BaseModel):
    """Schema for a sentiment analysis response."""

    text : str = Field(
        ...,
        description = "The original text that was analyzed."
    )
    
    sentiment : str = Field(
        ...,
        description = "The detected sentiment: positive, negative, or neutral."
    )

    confidence : float = Field(
        ...,
        ge = 0.0,
        le = 1.0,
        description = "Confidence score for the predicted sentiment (0-1)."
    )

    probabilities : Dict[str, float] = Field(
        ...,
        description = "Probability scores for each sentiment category."
    )

    model_config = ConfigDict(
        json_schema_extra = {
            "example" : {
                "text" : "I really enjoyed using this product, it exceeded my expectations!",
                "sentiment": "positive",
                "confidence": 0.92,
                "probabilities": {
                    "positive": 0.92,
                    "negative": 0.02,
                    "neutral": 0.06
                }
            }
        }
    )

class BatchSentimentResponse(BaseModel):
    """Schema for a batch sentiment analysis response."""

    results : List[SentimentResponse] = Field(
        ...,
        description = "List of sentiment analysis results for each input text."
    )