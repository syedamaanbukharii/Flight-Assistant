"""FastAPI application factory.

Wires logging, CORS, a request-id middleware and centralised exception
handlers around the API router. The application's object graph is built once
during the lifespan startup and exposed via ``app.state.container``; everything
is torn down cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from app.config import Settings, get_settings
from app.startup import build_container, build_scheduler, init_database
from utils.errors import AppError
from utils.logger import configure_logging, get_logger, get_request_id, set_request_id

_logger = get_logger("app.main")
_REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the container on startup and release it on shutdown."""
    settings: Settings = app.state.settings
    container = build_container(settings)
    await asyncio.to_thread(init_database, container)

    scheduler = build_scheduler(container)
    if scheduler is not None:
        scheduler.start()
        container.scheduler = scheduler

    app.state.container = container
    _logger.info("Application startup complete", extra={"env": settings.app_env})
    try:
        yield
    finally:
        await container.aclose()
        _logger.info("Application shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _assign_request_id(request: Request, call_next):
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        set_request_id(request_id)
        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response

    _register_exception_handlers(app)
    app.include_router(router)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            _logger.error(
                "Handled application error",
                extra={"code": exc.code, "detail": exc.message},
            )
        payload = exc.to_public_dict()
        payload["request_id"] = get_request_id()
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "request_validation_error",
                    "message": "The request payload is invalid.",
                    "details": _safe_validation_details(exc),
                },
                "request_id": get_request_id(),
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        _logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                },
                "request_id": get_request_id(),
            },
        )


def _safe_validation_details(exc: RequestValidationError) -> list[dict[str, str]]:
    """Summarise validation errors without echoing raw input values."""
    details: list[dict[str, str]] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        details.append({"field": location or "body", "message": error.get("msg", "")})
    return details


app = create_app()
