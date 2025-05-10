import os
import sys
import logging
import argparse
from pathlib import Path

# PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.model_loader import get_model
from app.models.prediction import predict_sentiment
from app.models.evaluation import evaluate_predictions
from data.datasets.download_datasets import download_and_prepare_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_on_dataset(dataset_name, split="test", num_samples=10):
    """
    Evaluate model on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to use (default: test)
        num_samples: Number of samples to use (None for all)
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating model on {dataset_name} ({split} split)")

    datasets = download_and_prepare_datasets([dataset_name])
    dataset_dict = datasets[dataset_name]
    dataset = dataset_dict["dataset"]

    from data.datasets.dataset_registry import get_dataset_config
    dataset_config = get_dataset_config(dataset_name)

    if not dataset_config:
        logger.error(f"No configuration found for dataset {dataset_name}")
        return None
    
    text_column = dataset_config.get("text_column", "text")
    label_column = dataset_config.get("label_column", "label")
    label_mapping = dataset_config.get("label_mapping", {0 : "negative", 1 : "neutral", 2 : "positive"})

    if split not in dataset:
        available_splits = list(dataset.keys())
        logger.warning(f"Split '{split}' not found. Available splits: {available_splits}")
        split = available_splits[0]
    
    samples = dataset[split]
    if num_samples:
        samples = samples.select(range(min(num_samples, len(samples))))
    
    # Extract texts and labels
    texts = samples[text_column]
    
    if dataset_name == "amazon_reviews_multi":
        true_labels = [label_mapping[sample["sentiment"]] for sample in samples]
    else:
        true_labels = [label_mapping[sample[label_column]] for sample in samples]
    
    predictions = predict_sentiment(texts)
    predicted_labels = [pred["sentiment"] for pred in predictions]

    metrics = evaluate_predictions(true_labels, predicted_labels)
    
    logger.info(f"Evaluation results for {dataset_name}:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (macro): {metrics['macro_f1']:.4f}")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    import json
    with open(results_dir / f"evaluation_{dataset_name}_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def main():
    """Main function for command line execution."""

    parser = argparse.ArgumentParser(description="Evaluate sentiment model on datasets")
    parser.add_argument("--dataset", default="tweet_eval", help = "Dataset to evaluate on")
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to use")

    args = parser.parse_args()

    evaluate_on_dataset(args.dataset, args.split, args.samples)

if __name__ == "__main__":
    main()