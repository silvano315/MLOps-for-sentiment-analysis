import logging
import time
import torch
import numpy as np
from torch.nn.functional import softmax
from typing import Optional, Any, Dict, List, Union

from app.models.model_loader import get_model, SentimentModel
from app.models.tokenization import tokenize_texts

logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Class for making predictions with the sentiment analysis model."""

    def __init__(self, model: Optional[SentimentModel] = None):
        """
        Initialize the predictor.
        
        Args:
            model: SentimentModel instance (if None, will be loaded)
        """

        self.model = model or get_model()

    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Perform sentiment analysis on the given text(s).
        
        Args:
            texts: A single text or list of texts for sentiment analysis
            
        Returns:
            List of dictionaries containing sentiment predictions for each text
        """

        start_time = time.time()

        if isinstance(texts, str):
            texts = [texts]
        
        encoded_inputs = tokenize_texts(texts, self.model.tokenizer)

        encoded_inputs = {k : v.to(self.model.device) for k, v in encoded_inputs.items()}

        # Inference
        try:
            with torch.no_grad():
                outputs = self.model.model(**encoded_inputs)
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

        predictions = []
        logits = outputs.logits.cpu().numpy()

        probabilities = softmax(torch.Tensor(logits), dim = 1).numpy()

        for i, (text, probs) in enumerate(zip(texts, probabilities)):
            predicted_class = int(np.argmax(probs))
            sentiment = self.model.id2label[predicted_class]

            result = {
                "text" : text,
                "sentiment" : sentiment,
                "confidence" : float(probs[predicted_class]),
                "probabilities" : {
                    label : float(probs[self.model.label2id[label]])
                    for label in self.model.id2label.values()
                }
            }
            predictions.append(result)

        elapsed_time = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed_time:.4f} seconds")

        return predictions

def predict_sentiment(texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Function to perform sentiment analysis.
    
    Args:
        texts: Text or list of texts to analyze
        
    Returns:
        List of prediction results
    """

    predictor = SentimentPredictor()
    return predictor.predict(texts)