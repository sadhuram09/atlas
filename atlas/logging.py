"""
atlas/logging.py

Structured logging via structlog.

Why structlog over standard logging?
  - Every log line is a dict → easy to query in Railway's log viewer
  - Context vars (task_id, agent) automatically added to every line
  - In development: pretty colored output
  - In production: JSON → grep / alerting / LangSmith

Usage:
    from atlas.logging import get_logger
    log = get_logger(__name__)
    log.info("task_started", task_id=task_id, agent="coder")
"""

import logging
import sys

import structlog
from structlog.types import EventDict

from atlas.config import settings


def _add_app_context(
    logger: logging.Logger,  # noqa: ARG001
    method_name: str,        # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject app name + environment into every log line."""
    event_dict["app"] = settings.app_name
    event_dict["env"] = settings.environment
    return event_dict


def configure_logging() -> None:
    """
    Call once at application startup (main.py lifespan).

    In development: timestamped, colored, human-readable.
    In production: JSON — one dict per line, parseable by Railway.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
    ]

    if settings.is_production:
        # JSON output — every line is a valid JSON object
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty colored output for local dev
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound logger. Call at module level: log = get_logger(__name__)"""
    return structlog.get_logger(name)
