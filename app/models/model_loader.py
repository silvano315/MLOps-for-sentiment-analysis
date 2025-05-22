import logging
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from app.utils.config import get_settings

logger = logging.getLogger(__name__)


class SentimentModel:
    """
    Class for loading and managing the sentiment analysis model.

    This class handles loading the pre-trained RoBERTa model from HuggingFace,
    and provides methods for tokenization and inference.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the sentiment model.

        Args:
            model_name: Name or path of the pre-trained model
            cache_dir: Directory to store the downloaded model
            device: Device to use for inference ('cpu', 'cuda', etc.)
        """

        settings = get_settings()
        self.model_name = model_name or settings.MODEL_NAME
        self.cache_dir = cache_dir or settings.MODEL_CACHE_DIR

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model()

        self._load_label_mapping()

    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""

        logger.info(f"Loading model: {self.model_name}")

        try:
            self.config = AutoConfig.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            if self._is_peft_model(self.model_name):
                base_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                self._load_peft_model(base_model_name, self.model_name)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir, low_cpu_mem_usage=False
                )
                self.model.to(self.device)

            if self._is_peft_model(self.model_name):
                tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            else:
                tokenizer_name = self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, cache_dir=self.cache_dir
            )

            logger.info("model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _load_label_mapping(self) -> None:
        """Load label mapping for the model."""
        # [From HugginFace] Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def _is_peft_model(self, model_name: str) -> bool:
        """Check if the model is a PEFT (LoRA) model."""
        try:
            from huggingface_hub import model_info
            info = model_info(model_name)
            return 'peft' in [tag for tag in info.tags if tag]
        except:
            return False

    def _load_peft_model(self, base_model_name: str, adapter_model_name: str):
        """Load a PEFT model with its base model."""
        from peft import PeftModel
        
        logger.info(f"Loading PEFT model: {adapter_model_name} with base: {base_model_name}")
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=False
        )
        
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_model_name,
            cache_dir=self.cache_dir
        )
        
        self.model.to(self.device)
        
        logger.info("PEFT model loaded successfully")

    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List available pre-trained models that can be used.

        Returns:
            Dictionary of model information
        """
        return {
            "cardiffnlp/twitter-roberta-base-sentiment-latest": {
                "name": "Twitter RoBERTa Base Sentiment",
                "description": "RoBERTa model fine-tuned on Twitter data for sentiment analysis",
                "type": "roberta",
                "languages": ["en"],
            },
        }

    def predict(self, texts: Union[str, List[str]]) -> None:  # List[Dict[str, Any]]:
        """
        Perform sentiment analysis on text(s).

        Args:
            texts: A single text or list of texts to analyze

        Returns:
            List of dictionaries containing sentiment predictions for each text
        """
        # This is a placeholder: implementation in app/model/prediction.py
        pass


_model_instance = None


def get_model() -> SentimentModel:
    """
    Get or create a singleton instance of the sentiment model.

    Returns:
        Sentiment model instance
    """

    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModel()
    return _model_instance
