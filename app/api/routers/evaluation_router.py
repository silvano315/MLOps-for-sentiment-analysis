import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Query

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/evaluate",
    summary="Evaluate model performance",
    description="Start a model evaluation on a specific dataset.",
)
async def evaluate_model(
    background_tasks: BackgroundTasks,
    dataset: str = Query("tweet_eval", description="Dataset to evaluate on"),
    split: str = Query("test", description="Dataset split to use"),
    samples: Optional[int] = Query(
        None, description="Number of samples to use (None for all)"
    ),
    model_name: Optional[str] = Query(None, description="Model name to evaluate"),
) -> Dict[str, Any]:
    """
    Trigger model evaluation in the background.

    Args:
        background_tasks: FastAPI background tasks
        dataset: Dataset to evaluate on
        split: Dataset split to use
        samples: Number of samples to use
        model_name: Model name to evaluate

    Returns:
        Status message
    """

    def run_evaluation():
        try:
            logger.info(
                f"Starting model evaluation on {dataset} ({split}) with samples={samples}"
            )

            env = os.environ.copy()
            if model_name:
                env["MODEL_NAME"] = model_name

            from scripts.evaluate_model import evaluate_on_dataset

            metrics = evaluate_on_dataset(dataset, split, samples, model_name)

            if metrics:
                logger.info(f"Evaluation completed successfully")
                logger.info(
                    f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}"
                )
            else:
                logger.error("Evaluation failed - no metrics returned")

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            import traceback

            traceback.print_exc()

    background_tasks.add_task(run_evaluation)

    return {
        "status": "evaluation_started",
        "message": f"Model evaluation started on {dataset} ({split})",
        "dataset": dataset,
        "split": split,
        "samples": samples,
        "model": model_name or "default",
    }
