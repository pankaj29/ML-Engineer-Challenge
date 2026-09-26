"""Structured logging with correlation IDs.

Plain English:
    In production, logs are read by machines before humans. So every log line
    is a single JSON object rather than free-form text. That lets you run
    queries like "show me every request from user X that took over a second".

    A *correlation ID* is a random id generated once per HTTP request. It is
    attached to every log line produced while handling that request, returned
    to the client in the ``X-Correlation-ID`` header, and stored on the
    inference record in Postgres. When a user reports "my request failed at
    10:42", that one id pulls up the complete story across API, worker and
    database.

The correlation id lives in a :class:`contextvars.ContextVar`, which is the
async-safe equivalent of a thread-local: each concurrent request gets its own
value even though they share a thread.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from pythonjsonlogger import jsonlogger

# Request-scoped context. Set by LoggingMiddleware, read by the log filter.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")

# Attributes the stdlib puts on every LogRecord. We skip them when copying
# "extra" fields into the JSON output, otherwise every line would be enormous.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


def new_correlation_id() -> str:
    """Generate a fresh correlation id (32-char hex, no dashes)."""
    return uuid.uuid4().hex


def get_correlation_id() -> str:
    """Return the current request's correlation id, or "" outside a request."""
    return correlation_id_var.get()


# Inbound ids come from a client header. The database columns are 64 wide,
# and anything that is not a plain token would be written into every log line.
_VALID_CORRELATION_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


def bind_correlation_id(cid: str | None = None) -> str:
    """Set the correlation id for the current context and return it.

    A missing or malformed id is replaced with a fresh one.
    """
    if not cid or not _VALID_CORRELATION_ID.fullmatch(cid):
        cid = new_correlation_id()
    correlation_id_var.set(cid)
    return cid


class CorrelationIdFilter(logging.Filter):
    """Copy the context-local correlation id onto every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = getattr(record, "correlation_id", None) or correlation_id_var.get()
        record.user_id = getattr(record, "user_id", None) or user_id_var.get()
        return True


class JsonFormatter(jsonlogger.JsonFormatter):
    """Emit one JSON object per log line, with stable top-level keys."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["event"] = record.getMessage()
        log_record.setdefault("correlation_id", correlation_id_var.get())
        log_record.setdefault("user_id", user_id_var.get())
        log_record["source"] = f"{record.module}:{record.funcName}:{record.lineno}"
        # Drop empty/null keys so downstream log stores are not full of nulls
        # (`taskName` in particular is None for every non-asyncio log line).
        for key in [k for k, v in log_record.items() if v is None or v == ""]:
            log_record.pop(key, None)


class ConsoleFormatter(logging.Formatter):
    """Human-friendly single-line format for local development.

    Shows the first 8 characters of the correlation id, which is plenty to
    eyeball which lines belong to the same request while tailing a terminal.
    """

    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "") or ""
        cid_part = f"[{cid[:8]}] " if cid else ""
        level = record.levelname
        if self.use_color:
            level = f"{self._COLORS.get(level, '')}{level:<8}{self._RESET}"
        else:
            level = f"{level:<8}"
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        base = f"{ts} {level} {cid_part}{record.name} - {record.getMessage()}"

        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_RECORD_FIELDS
            and k not in ("correlation_id", "user_id")
            and not k.startswith("_")
            and v is not None
        }
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(
    level: str = "INFO",
    log_format: str = "json",
    *,
    quiet_loggers: tuple[str, ...] = ("uvicorn.access", "botocore", "urllib3", "asyncio"),
) -> None:
    """Configure the root logger once, at process start.

    Args:
        level: Minimum level to emit (``"DEBUG"`` ... ``"CRITICAL"``).
        log_format: ``"json"`` for production, ``"console"`` for local dev.
        quiet_loggers: Third-party loggers turned down to WARNING so they do
            not drown out our own events. ``uvicorn.access`` in particular is
            redundant: our own middleware logs richer access records.
    """
    root = logging.getLogger()
    # Remove existing handlers so repeated calls (e.g. in tests) do not
    # produce duplicate log lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JsonFormatter("%(timestamp)s %(level)s %(logger)s %(event)s"))
    else:
        handler.setFormatter(ConsoleFormatter())
    handler.addFilter(CorrelationIdFilter())

    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "logging_configured", extra={"level": level.upper(), "format": log_format}
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Thin wrapper for import consistency."""
    return logging.getLogger(name)


__all__ = [
    "CorrelationIdFilter",
    "bind_correlation_id",
    "configure_logging",
    "correlation_id_var",
    "get_correlation_id",
    "get_logger",
    "new_correlation_id",
    "user_id_var",
]
