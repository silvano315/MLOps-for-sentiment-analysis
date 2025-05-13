from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Any, Dict, List


class SentimentRequest(BaseModel):
    """Schema for a single sentiment analysis request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="The text to analyze for sentiment.",
    )

    @field_validator("text")
    def text_must_not_be_empty(cls, v):
        """Validate that text is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Text must not be empty or whitespace only")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "I really enjoyed using this product, it exceeded my expectations!"
            }
        }
    )


class BatchSentimentRequest(BaseModel):
    """Schema for a batch sentiment analysis request."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze for sentiment.",
    )

    @field_validator("texts")
    def texts_must_not_be_empty(cls, v):
        """Validate that each text is not empty."""
        if not v:
            raise ValueError("Texts list must not be empty")

        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} must not be empty or whitespace only"
                )

        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "I really enjoyed using this product, it exceeded my expectations!",
                    "The customer service was terrible, I had to wait for hours.",
                    "It's an average product, nothing special but gets the job done.",
                ]
            }
        }
    )
