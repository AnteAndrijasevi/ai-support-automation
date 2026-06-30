import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.api.routes import health
from app.config import get_settings
from app.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("app.request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger("app").info("starting up", extra={"app_env": settings.app_env})
    yield
    logging.getLogger("app").info("shutting down")


app = FastAPI(
    title="AI Support Automation",
    description=(
        "Backend service that triages incoming customer support tickets with an LLM: "
        "classification (category, urgency, sentiment), a drafted reply, and a "
        "confidence flag for human review. All sample data is synthetic."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request completed",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


app.include_router(health.router)
