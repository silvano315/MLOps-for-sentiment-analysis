from data.datasets.dataset_registry import get_dataset_config, map_amazon_stars


def test_dataset_registry_contains_expected_datasets():
    """Test that the dataset registry contains the expected datasets."""

    from data.datasets.dataset_registry import DATASET_REGISTRY

    assert "tweet_eval" in DATASET_REGISTRY
    assert "mteb/amazon_reviews_multi" in DATASET_REGISTRY


def test_get_dataset_config_returns_correct_config():
    """Test that get_dataset_config returns the correct configuration."""

    config = get_dataset_config("tweet_eval")

    assert config is not None
    assert config["name"] == "tweet_eval"
    assert config["config"] == "sentiment"
    assert "label_mapping" in config


def test_map_amazon_stars_functions_correctly():
    """Test that map_amazon_stars maps stars correctly."""

    assert map_amazon_stars(0) == 0
    assert map_amazon_stars(1) == 0
    assert map_amazon_stars(2) == 1
    assert map_amazon_stars(3) == 2
    assert map_amazon_stars(4) == 2
