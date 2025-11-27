# src/common/logging_utils.py

"""
Privacy-preserving logging utilities for the SecureLLM system.

This module ensures:
- No raw prompt or decrypted content is ever logged.
- Logs are structured as key=value pairs.
- Only metadata (request_id, tenant_id, worker_id, timing info) is logged.
- Logging level is controlled by config.LOG_LEVEL.

We use a very lightweight wrapper on top of Python's standard logging module.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Dict

from common.config import config


# -------------------------------------------------------------------------
# Logger Initialization
# -------------------------------------------------------------------------

def _initialize_logger() -> logging.Logger:
    """
    Initializes a logger with stdout handler and configurable log level.
    Logs are formatted as: timestamp level message key=value key=value ...
    """
    logger = logging.getLogger("securellm")
    logger.setLevel(config.LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)

    # Avoid multiple handlers if this is re-imported
    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger


_logger = _initialize_logger()


# -------------------------------------------------------------------------
# Sanitized logging function
# -------------------------------------------------------------------------

def log_event(
    message: str,
    *,
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
    level: str = "info",
):
    """
    Logs a sanitized, structured message with no sensitive data.

    Rules:
    - Do NOT log raw prompts or decrypted plaintext.
    - Only log metadata: request_id, tenant_id, worker_id, timing, error codes.
    - extra={} can include additional safe metadata (never content).

    Examples:
        log_event(
            "worker_started",
            worker_id="w3",
            extra={"state": "READY"}
        )

        log_event(
            "inference_completed",
            request_id="abc123",
            tenant_id="t1",
            extra={"latency_ms": "1234"},
        )
    """

    # Build structured key=value pairs
    fields = []

    if request_id:
        fields.append(f"request_id={request_id}")

    if tenant_id:
        if config.ALLOW_HOST_LOGGING_OF_IDS:
            fields.append(f"tenant_id={tenant_id}")

    if worker_id:
        fields.append(f"worker_id={worker_id}")

    if error:
        fields.append(f"error={error}")

    # Add additional safe metadata
    if extra:
        for k, v in extra.items():
            fields.append(f"{k}={v}")

    # Construct final message
    full_message = f"{message} " + " ".join(fields)

    # Dispatch to logger
    level = level.lower()
    if level == "info":
        _logger.info(full_message)
    elif level == "warning":
        _logger.warning(full_message)
    elif level == "error":
        _logger.error(full_message)
    else:
        _logger.debug(full_message)
