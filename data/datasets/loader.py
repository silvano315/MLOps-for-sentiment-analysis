import logging
from datasets import load_datasets, Dataset, DatasetDict
from typing import Optional, Any, Dict, Tuple, Union

from app.utils.config import get_settings

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Class for loading and managing datasets for sentiment analysis."""

    def load_primary_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Load the primary dataset (TweetEval).
        
        Returns:
            Dataset or DatasetDict object
        """

        try:
            logger.info(f"Loading primary dataset: {self.settings.PRIMARY_DATASET}")
            dataset = load_datasets(
                self.settings.PRIMARY_DATASET,
                self.settings.PRIMARY_DATASET_CONFIG
            )
            logger.info("Primary dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading primary dataset: {str(e)}")
            raise RunTimeError(f"Failed to load primary dataset: {str(e)}")

    def load_secondary_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Load the secondary dataset (Amazon Reviews).
        
        Returns:
            Dataset or DatasetDict object
        """

        try:
            logger.info(f"Loading secondary dataset: {self.settings.SECONDARY_DATASET}")
            dataset = load_dataset(
                self.settings.SECONDARY_DATASET,
                self.settings.SECONDARY_DATASET_CONFIG
            )
            logger.info(f"Secondary dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading secondary dataset: {str(e)}")
            raise RuntimeError(f"Failed to load secondary dataset: {str(e)}")

    def get_dataset_sample(
        self,
        dataset : Union[Dataset, DatasetDict],
        split : str = "train",
        sample_size : int = 5
    ) -> Dataset:
        """
        Get a small sample from a dataset for quick testing.
        
        Args:
            dataset: Dataset to sample from
            split: Dataset split to use
            sample_size: Number of samples to retrieve
            
        Returns:
            Dataset containing samples
        """

        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available splits: {splits}")
                split = splits[0]
            return dataset[split].select(range(min(sample_size, len(dataset[split]))))
        else:
            return dataset.select(range(min(sample_size, len(dataset))))