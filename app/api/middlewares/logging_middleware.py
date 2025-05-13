import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, log details, and pass to the next middleware.

        Args:
            request: The incoming request
            call_next: The next middleware to call

        Returns:
            The response from the next middleware
        """

        start_time = time.time()

        request_id = request.headers.get("X-Request-ID", "")
        client_ip = request.client.host if request.client else "unknown"
        request_path = request.url.path
        request_method = request.method

        logger.info(
            f"Request started: {request_method} {request_path} "
            f"from {client_ip} [request_id = {request_id}]"
        )

        # Process request through the rest of the application
        try:
            response = await call_next(request)

            process_time = time.time() - start_time

            logger.info(
                f"Request completed: {request_method} {request_path} "
                f"- Status: {response.status_code} - "
                f"Duration: {process_time:.4f}s [request_id={request_id}]"
            )

            response.headers["X-Process-Time"] = str(process_time)

            return response
        except Exception as e:
            logger.error(
                f"Request failed: {request_method} {request_path} "
                f"- Error: {str(e)} [request_id = {request_id}]"
            )
            raise
