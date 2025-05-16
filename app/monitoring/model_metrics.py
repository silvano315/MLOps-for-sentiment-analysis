import logging
from typing import Any, Dict

from app.monitoring.metrics import (
    MODEL_ACCURACY,
    MODEL_CONFUSION_MATRIX,
    MODEL_F1,
    MODEL_PRECISION,
    MODEL_RECALL,
)

logger = logging.getLogger(__name__)


def record_model_evaluation_metrics(
    evaluation_results: Dict[str, Any],
    model_name: str = "default",
    dataset: str = "unknown",
    split: str = "test",
) -> None:
    """
    Record model evaluation metrics to Prometheus.

    Args:
        evaluation_results: Dictionary of evaluation metrics from evaluate_predictions
        model_name: Name of the model being evaluated
        dataset: Name of the dataset used for evaluation
        split: Dataset split used (train, val, test)
    """
    try:
        MODEL_ACCURACY.labels(model_name=model_name, dataset=dataset, split=split).set(
            evaluation_results["accuracy"]
        )

        # Macro-averaged metrics
        MODEL_PRECISION.labels(
            model_name=model_name,
            dataset=dataset,
            split=split,
            class_label="macro",
            average="macro",
        ).set(evaluation_results["macro_precision"])

        MODEL_RECALL.labels(
            model_name=model_name,
            dataset=dataset,
            split=split,
            class_label="macro",
            average="macro",
        ).set(evaluation_results["macro_recall"])

        MODEL_F1.labels(
            model_name=model_name,
            dataset=dataset,
            split=split,
            class_label="macro",
            average="macro",
        ).set(evaluation_results["macro_f1"])

        # Class-specific metrics
        for class_name, metrics in evaluation_results["class_metrics"].items():
            MODEL_PRECISION.labels(
                model_name=model_name,
                dataset=dataset,
                split=split,
                class_label=class_name,
                average="none",
            ).set(metrics["precision"])

            MODEL_RECALL.labels(
                model_name=model_name,
                dataset=dataset,
                split=split,
                class_label=class_name,
                average="none",
            ).set(metrics["recall"])

            MODEL_F1.labels(
                model_name=model_name,
                dataset=dataset,
                split=split,
                class_label=class_name,
                average="none",
            ).set(metrics["f1"])

        confusion_matrix = evaluation_results["confusion_matrix"]
        class_names = list(evaluation_results["class_metrics"].keys())

        for i, true_label in enumerate(class_names):
            for j, pred_label in enumerate(class_names):
                MODEL_CONFUSION_MATRIX.labels(
                    model_name=model_name,
                    dataset=dataset,
                    split=split,
                    true_label=true_label,
                    predicted_label=pred_label,
                ).set(confusion_matrix[i][j])

        logger.info(
            f"Recorded model evaluation metrics for {model_name} on {dataset} ({split})"
        )
    except Exception as e:
        logger.error(f"Failed to record model evaluation metrics: {str(e)}")
