from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    true_labels: List[str], predicted_labels: List[str]
) -> Dict[str, Any]:
    """
    Evaluate model predictions against true labels.

    Args:
        true_labels: List of true sentiment labels
        predicted_labels: List of predicted sentiment labels

    Returns:
        Dictionary containing evaluation metrics
    """

    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    if isinstance(true_labels[0], str):
        true_indices = [label_map.get(label, 1) for label in true_labels]
    else:
        true_indices = true_labels

    if isinstance(predicted_labels[0], str):
        pred_indices = [label_map.get(label, 1) for label in predicted_labels]
    else:
        pred_indices = predicted_labels

    accuracy = accuracy_score(true_indices, pred_indices)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_indices, pred_indices, average=None, labels=[0, 1, 2]
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_indices, pred_indices, average="macro"
    )

    cm = confusion_matrix(true_indices, pred_indices, labels=[0, 1, 2])

    class_metrics = {}
    classes = ["negative", "neutral", "positive"]

    for i, cls in enumerate(classes):
        class_metrics[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    return results
