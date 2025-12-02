# src/common/logging_utils.py

"""
Privacy-preserving logging utilities for the SecureLLM system.

This module ensures:
- No raw prompt or decrypted content is ever logged.
- Logs are structured as key=value pairs.
- Only metadata (request_id, tenant_id, worker_id, timing info) is logged.
- Logging level is controlled by config.LOG_LEVEL.
- STRICT_NO_LOGGING_MODE and ALLOW_HOST_LOGGING_OF_IDS from config
  act as global privacy levers.

We use a very lightweight wrapper on top of Python's standard logging module.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Dict, Any

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

    # Allow LOG_LEVEL to be a string like "INFO", "DEBUG", etc.
    try:
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    except Exception:
        logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)

    # Avoid multiple handlers if this is re-imported
    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger


_logger = _initialize_logger()


# -------------------------------------------------------------------------
# Privacy-aware sanitization
# -------------------------------------------------------------------------

# Keys that should never be logged (defense in depth)
SENSITIVE_KEYS = {
    "prompt",
    "full_prompt",
    "raw_prompt",
    "input_text",
    "user_message",
    "response",
    "raw_output",
    "raw_response",
    "rag_context",
    "plaintext",
    "decrypted",
    "tokens",
    "token_ids",
    "embedding",
    "embeddings",
}


def _sanitize_extra(extra: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Sanitize extra metadata fields before logging.

    Rules:
    - Drop known sensitive keys (SENSITIVE_KEYS) always.
    - In STRICT_NO_LOGGING_MODE:
        - Drop tenant_id if ALLOW_HOST_LOGGING_OF_IDS is False.
        - Drop very long string values (heuristic against content leakage).
    - Always coerce values to short strings.
    """
    if not extra:
        return {}

    cleaned: Dict[str, str] = {}

    for key, value in extra.items():
        # Normalize key as string
        k = str(key)

        # 1) Never log obviously sensitive keys
        if k in SENSITIVE_KEYS:
            continue

        # 2) Additional constraints in strict mode
        if config.STRICT_NO_LOGGING_MODE:
            # tenant_id may be present in extra; enforce global policy
            if k == "tenant_id" and not config.ALLOW_HOST_LOGGING_OF_IDS:
                continue

            # Heuristic: avoid logging very long strings that might contain content
            if isinstance(value, str) and len(value) > 256:
                continue

        # 3) Coerce value to string for consistent key=value logging
        cleaned[k] = str(value)

    return cleaned


def _level_to_int(level: str) -> int:
    lvl = level.lower()
    if lvl == "info":
        return logging.INFO
    if lvl == "warning":
        return logging.WARNING
    if lvl == "error":
        return logging.ERROR
    if lvl == "debug":
        return logging.DEBUG
    return logging.INFO


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
    extra: Optional[Dict[str, Any]] = None,
    level: str = "info",
):
    """
    Logs a sanitized, structured message with no sensitive data.

    Rules:
    - Do NOT log raw prompts or decrypted plaintext.
    - Only log metadata: request_id, tenant_id, worker_id, timing, error codes.
    - extra={} can include additional safe metadata (never content).
    - STRICT_NO_LOGGING_MODE + ALLOW_HOST_LOGGING_OF_IDS control how much
      metadata is allowed.

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
            extra={"latency_ms": 1234},
        )
    """

    fields = []

    # Basic identifiers (respect global flags)
    if request_id:
        fields.append(f"request_id={request_id}")

    if tenant_id:
        if config.ALLOW_HOST_LOGGING_OF_IDS:
            fields.append(f"tenant_id={tenant_id}")
        elif not config.STRICT_NO_LOGGING_MODE:
            # Non-strict mode: you could decide to allow this, but we keep it off
            pass

    if worker_id:
        fields.append(f"worker_id={worker_id}")

    if error:
        # Error strings are typically short; if not, they will still be logged,
        # but they should not contain user content.
        fields.append(f"error={error}")

    # Sanitize and append extra metadata
    safe_extra = _sanitize_extra(extra)
    for k, v in safe_extra.items():
        fields.append(f"{k}={v}")

    # Mark strict mode in logs if enabled (useful for debugging behavior)
    if config.STRICT_NO_LOGGING_MODE:
        fields.append("strict_no_logging=True")

    full_message = f"{message} " + " ".join(fields)

    _logger.log(_level_to_int(level), full_message)