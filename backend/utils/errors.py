"""Typed exception hierarchy for the application.

Every exception carries an HTTP status code and a stable machine-readable
``code`` so the API layer can translate failures into clean JSON responses
without leaking stack traces or internal details to clients.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all application errors.

    Args:
        message: Human-readable, client-safe description of the error.
        status_code: HTTP status code the API layer should return.
        code: Stable machine-readable error identifier.
        detail: Optional structured context (never exposed verbatim to clients
            unless it is known to be safe).
    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        detail: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        self.detail = detail

    def to_public_dict(self) -> dict[str, Any]:
        """Return a client-safe representation of the error."""
        return {"error": {"code": self.code, "message": self.message}}


class ConfigurationError(AppError):
    """Raised when required configuration is missing or invalid."""

    status_code = 500
    code = "configuration_error"


class NotConfiguredError(AppError):
    """Raised when a feature is requested but its integration is not configured."""

    status_code = 503
    code = "not_configured"


class ValidationAppError(AppError):
    """Raised for domain-level validation failures."""

    status_code = 400
    code = "validation_error"


class ExternalServiceError(AppError):
    """Raised when an upstream provider (HTTP API) fails."""

    status_code = 502
    code = "upstream_error"


class LLMError(AppError):
    """Raised when the language model call fails or returns unusable output."""

    status_code = 502
    code = "llm_error"


class ToolExecutionError(AppError):
    """Raised when a planner tool fails during execution."""

    status_code = 500
    code = "tool_execution_error"
