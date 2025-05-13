import argparse
import logging
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from app.utils.config import get_settings
from data.datasets.dataset_registry import (
    DATASET_REGISTRY,
    get_dataset_config,
    map_amazon_stars,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download a dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset
        config_name: Configuration name
        save_path: Path to save the dataset

    Returns:
        Dictionary with dataset info and paths
    """

    logger.info(
        f"Downloading dataset {dataset_name}"
        + (f" ({config_name})" if config_name else "")
    )

    dataset_config = get_dataset_config(dataset_name)
    if not dataset_config and dataset_name in DATASET_REGISTRY:
        dataset_config = DATASET_REGISTRY[dataset_name]

    if not config_name and dataset_config and "config" in dataset_config:
        config_name = dataset_config["config"]

    try:
        dataset = load_dataset(dataset_name, config_name)

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            dataset_info = {
                "name": dataset_name,
                "config": config_name,
                "splits": (
                    list(dataset.keys())
                    if isinstance(dataset, DatasetDict)
                    else ["train"]
                ),
                "num_rows": (
                    {split: len(dataset[split]) for split in dataset.keys()}
                    if isinstance(dataset, DatasetDict)
                    else {"train": len(dataset)}
                ),
                "features": (
                    str(next(iter(dataset.values())).features)
                    if isinstance(dataset, DatasetDict)
                    else str(dataset.features)
                ),
                "path": save_path,
            }

            info_path = os.path.join(
                save_path,
                f"{dataset_name}_{config_name if config_name else 'default'}_info.json",
            )
            with open(info_path, "w") as f:
                import json

                json.dump(dataset_info, f, indent=2)

            logger.info(f"Dataset info saved to {info_path}")

            return {"dataset": dataset, "info": dataset_info, "path": save_path}

        return {
            "dataset": dataset,
            "info": {"name": dataset_name, "config": config_name},
        }

    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
        raise RuntimeError(f"Failed to download dataset {dataset_name}: {str(e)}")


def preprocess_dataset(dataset: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Preprocess a dataset based on its configuration.

    Args:
        dataset: Dataset dictionary returned by download_dataset
        dataset_name: Name of the dataset

    Returns:
        Processed dataset dictionary
    """

    logger.info(f"Preprocessing dataset {dataset_name}")
    dataset_obj = dataset["dataset"]
    dataset_config = get_dataset_config(dataset_name)

    if not dataset_config:
        logger.warning(f"No preprocessing configuration found for {dataset_name}")
        return dataset

    # Apply preprocessing
    if dataset_name == "amazon_reviews_multi" and "preprocessing" in dataset_config:
        if dataset_config["preprocessing"] == "map_amazon_stars":
            logger.info("Applying Amazon stars mapping")

            def map_stars(example):
                example["sentiment"] = map_amazon_stars(
                    example[dataset_config["label_column"]]
                )
                return example

            if isinstance(dataset_obj, DatasetDict):
                processed_dataset = {}
                for split, data in dataset_obj.items():
                    processed_dataset[split] = data.map(map_stars)
                dataset["dataset"] = DatasetDict(processed_dataset)
            else:
                dataset["dataset"] = dataset_obj.map(map_stars)

    if "info" in dataset:
        dataset["info"]["preprocessed"] = True

    return dataset


def download_and_prepare_datasets(
    dataset_names: Optional[List[str]] = None, save_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Download and prepare multiple datasets.

    Args:
        dataset_names: List of dataset names to download, or None for default datasets
        save_dir: Directory to save datasets

    Returns:
        Dictionary of datasets
    """

    settings = get_settings()

    if not dataset_names:
        dataset_names = [settings.PRIMARY_DATASET, settings.SECONDARY_DATASET]

    if not save_dir:
        save_dir = os.path.join(os.getcwd(), "data", "datasets", "cached")

    os.makedirs(save_dir, exist_ok=True)

    datasets = {}

    for dataset_name in dataset_names:
        try:
            dataset_config = get_dataset_config(dataset_name)
            config_name = dataset_config.get("config") if dataset_config else None

            dataset_dir = os.path.join(save_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            dataset_dict = download_dataset(dataset_name, config_name, dataset_dir)

            preprocess_dataset_ = preprocess_dataset(dataset_dict, dataset_name)

            datasets[dataset_name] = preprocess_dataset_

            logger.info(f"Successfully prepared dataset {dataset_name}")
        except Exception as e:
            logger.error(f"Error preparing dataset {dataset_name}: {str(e)}")

    return datasets


def main():
    """Main function for command line execution."""

    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for sentiment analysis"
    )
    parser.add_argument(
        "--datasets", nargs="+", help="List of dataset names to download"
    )
    parser.add_argument("--save-dir", help="Directory to save datasets")

    args = parser.parse_args()

    download_and_prepare_datasets(args.datasets, args.save_dir)


if __name__ == "__main__":
    main()
