import platform
import time
from typing import Any, Dict

import psutil
import torch
from fastapi import APIRouter

from app.utils.config import get_settings

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Check the health status of the API and its components.",
)
async def health_check() -> Dict[str, Any]:
    """
    Perform a health check of the API and its components.

    Returns:
        Health status information
    """

    settings = get_settings()

    model_status = "available"
    model_error = None

    try:
        from app.models.model_loader import get_model

        model = get_model()
    except Exception as e:
        model_status = "unavailable"
        model_error = str(e)

    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.API_VERSION,
        "model": {
            "name": settings.MODEL_NAME,
            "status": model_status,
            "error": model_error,
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__ if torch else "not installed",
            "gpu_available": torch.cuda.is_available() if torch else False,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent_used": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent_used": disk.percent,
            },
        },
    }


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the API is ready to serve requests.",
)
async def readiness_check() -> Dict[str, str]:
    """
    Check if the service is ready to serve requests.

    Returns:
        Readiness status
    """

    return {"status": "ready"}


@router.get("/live", summary="Liveness check", description="Check if the API is alive")
async def liveness_check() -> Dict[str, str]:
    """
    Check if the service is alive.

    Returns:
        Liveness status
    """

    return {"status": "alive"}
