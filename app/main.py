import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.routes import health, tickets
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


class RequestLoggingMiddleware:
    """Pure ASGI middleware (not BaseHTTPMiddleware) so it doesn't run each
    request in a separate task -- simpler, avoids known BaseHTTPMiddleware
    issues with streaming/background tasks, and keeps coverage tooling able
    to see straight through it."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request completed",
            extra={
                "path": scope["path"],
                "method": scope["method"],
                "status_code": status_code,
                "latency_ms": latency_ms,
            },
        )


app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.error("database error", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable"})


app.include_router(health.router)
app.include_router(tickets.router)
