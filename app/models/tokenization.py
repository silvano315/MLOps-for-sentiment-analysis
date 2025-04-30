import torch
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Union

def tokenize_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of texts using the provided tokenizer.
    
    Args:
        texts: List of texts to tokenize
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy ('max_length', 'longest', etc.)
        truncation: Whether to truncate sequences longer than max_length
        return_tensors: Type of tensors to return ('pt' for PyTorch)
        
    Returns:
        Dictionary containing tokenized inputs
    """
    
    return tokenizer(
        texts,
        max_length = max_length,
        padding = padding,
        truncation = truncation,
        return_tensors = return_tensors
    )