import logging
from fastapi import Header, Depends, Request, HTTPException
from typing import Optional, Callable, Any, Dict

from app.utils.config import get_settings

logger = logging.getLogger(__name__)


async def get_api_key(
    api_key: str = Header(None, description="API key for authentication")
) -> str:
    """
    Validate the API key (TBD).

    Args:
        api_key: API key from request header

    Returns:
        Validated API key
    """

    # TO BE IMPLEMENTED
    return api_key


async def get_request_if(
    x_request_id: Optional[str] = Header(None, description="Unique request ID")
) -> str:
    """
    Get or generate a unique request ID.

    Args:
        x_request_id: Request ID from header

    Returns:
        Request ID
    """

    if not x_request_id:
        import uuid

        x_request_id = str(uuid.uuid4())
    return x_request_id
