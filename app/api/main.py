import time
import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

from app.api.routers import sentiment_router, health_router
from app.api.middlewares.logging_middleware import LoggingMiddleware
from app.utils.config import get_settings

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title = settings.API_TITLE,
    description = settings.API_DESCRIPTION,
    version = settings.API_VERSION,
    docs_url = "/docs",
    redoc_url = "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.add_middleware(LoggingMiddleware)

app.include_router(health_router.router, tags = ["Health"])
app.include_router(sentiment_router.router, prefix = "/api/v1", tags = ["Sentiment"])

@app.lifespan("startup")
async def startup_event():
    """
    Execute operations on application startup.
    """

    logger.info("Starting up sentiment analysis API")
    start_time = time.time()

    from app.models.model_loader import get_model
    get_model()

    logger.info(f"Startup completed in {time.time() - start_time:.2f} seconds")

@app.lifespan("shutdown")
async def shutdown_event():
    """
    Execute operations on application shutdown.
    """
    logger.info("Shutting down sentiment analysis API")