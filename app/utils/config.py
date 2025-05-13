from functools import lru_cache

from config.settings import Settings


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings using lru_cache for improved performance.

    Returns:
        Application settings object
    """
    return Settings()
