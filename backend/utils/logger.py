"""Structured JSON logging.

Logs are emitted as single-line JSON objects carrying a timestamp, level,
service name, message, the active request id, and any extra fields supplied by
the caller. A :class:`contextvars.ContextVar` propagates the request id across
async calls so every log line within a request can be correlated.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

# Standard ``LogRecord`` attributes we never want to duplicate into the JSON
# payload; anything outside this set is treated as a caller-supplied extra.
_RESERVED_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName",
    }
)


def set_request_id(request_id: str) -> None:
    """Bind ``request_id`` to the current execution context."""
    _request_id_ctx.set(request_id)


def get_request_id() -> str:
    """Return the request id bound to the current execution context."""
    return _request_id_ctx.get()


class JsonFormatter(logging.Formatter):
    """Render :class:`logging.LogRecord` instances as compact JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - logging API
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "service": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload.setdefault(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging to emit structured JSON to stdout.

    Idempotent: repeated calls replace existing handlers rather than stacking
    them, which keeps test runs and reloads clean.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Let uvicorn's loggers flow through our formatter instead of their own.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(service: str) -> logging.Logger:
    """Return a logger namespaced by ``service`` (used as the ``service`` field)."""
    return logging.getLogger(service)
