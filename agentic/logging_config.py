"""Structured JSON logging for UDA-Hub.

All log entries are JSON-formatted with consistent fields:
timestamp, agent, action, and optional ticket_id + details.
"""
import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """Formats log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "agent": getattr(record, "agent", record.name),
            "action": getattr(record, "action", "log"),
        }

        ticket_id = getattr(record, "ticket_id", None)
        if ticket_id:
            log_entry["ticket_id"] = ticket_id

        details = getattr(record, "details", None)
        if details:
            log_entry["details"] = details

        log_entry["message"] = record.getMessage()
        return json.dumps(log_entry)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured JSON output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance. Call setup_logging() first."""
    return logging.getLogger(name)


def log_structured(
    logger: logging.Logger,
    message: str,
    *,
    agent: str = "",
    action: str = "",
    ticket_id: str = "",
    details: dict | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit a structured JSON log entry."""
    extra = {
        "agent": agent or logger.name,
        "action": action,
    }
    if ticket_id:
        extra["ticket_id"] = ticket_id
    if details:
        extra["details"] = details
    logger.log(level, message, extra=extra)
