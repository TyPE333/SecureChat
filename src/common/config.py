# src/common/config.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Self


ENV_PREFIX = "SECURELLM_"


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(ENV_PREFIX + name)
    return value if value is not None else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(ENV_PREFIX + name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(ENV_PREFIX + name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(ENV_PREFIX + name)
    if value is None:
        return default
    value_lower = value.lower()
    if value_lower in ("1", "true", "yes", "y", "on"):
        return True
    if value_lower in ("0", "false", "no", "n", "off"):
        return False
    return default


@dataclass
class Config:
    """
    Central configuration for the secure LLM inference system.

    Defaults can be overridden via environment variables:
    - Prefix: SECURELLM_
    - Example: SECURELLM_WORKER_POOL_MIN_SIZE=2
    """
    WORKER_SECRET_KEY: bytes = b'0123456789ABCDEF0123456789ABCDEF'

    # --- Worker Pool / Lifecycle ---
    WORKER_POOL_MIN_SIZE: int = 1
    WORKER_POOL_MAX_SIZE: int = 4
    WORKER_IDLE_TIMEOUT_MS: int = 60_000
    WORKER_START_RETRIES: int = 2
    WORKER_HEALTHCHECK_INTERVAL_MS: int = 10_000

    # --- Timeouts & Deadlines ---
    HTTP_REQUEST_TIMEOUT_MS: int = 30_000
    ORCHESTRATOR_RPC_TIMEOUT_MS: int = 25_000
    MODEL_LOAD_TIMEOUT_MS: int = 10_000
    INFERENCE_TIMEOUT_MS: int = 15_000
    STREAM_IDLE_TIMEOUT_MS: int = 5_000
    RAG_TIMEOUT_MS: int = 2_000

    # --- Model / Inference ---
    MODEL_NAME: str = "llm-int8"
    MODEL_QUANTIZATION_MODE: str = "INT8"  # one of: FP16, INT8, INT4
    MAX_PROMPT_TOKENS: int = 2_048
    MAX_GENERATION_TOKENS: int = 512
    ENABLE_FLASH_ATTENTION: bool = True

    # --- RAG / Vector Retrieval ---
    RAG_ENABLED: bool = True
    RAG_FALLBACK_ENABLED: bool = True
    RAG_TOP_K: int = 4
    EMBEDDING_MODEL_NAME: str = "bge-small"
    VECTOR_DB_BACKEND: str = "faiss"  # or "qdrant"
    RAG_MAX_CONTEXT_TOKENS: int = 1_024

    # --- Security / Encryption ---
    ENCRYPTION_ALGO: str = "AES-GCM"
    KEY_ROTATION_INTERVAL_MIN: int = 60
    ALLOW_HOST_LOGGING_OF_IDS: bool = True
    STRICT_NO_LOGGING_MODE: bool = False

    # --- Metrics / Logging ---
    METRICS_ENABLED: bool = True
    METRICS_BACKEND: str = "stdout"  # or "prometheus"
    METRICS_SAMPLING_RATE: float = 1.0
    LOG_LEVEL: str = "INFO"

    # --- Multi-tenancy / Limits ---
    MAX_TENANTS: int = 10
    TENANT_RATE_LIMIT_QPS: int = 5
    TENANT_MAX_CONCURRENT_REQUESTS: int = 3

    # --- Region Mappings (simple default) ---
    REGION_TO_DC: Dict[str, str] = field(
        default_factory=lambda: {
            "us-west-1": "cluster-a",
            "us-east-1": "cluster-b",
        }
    )

    @classmethod
    def from_env(cls) -> Self:
        """
        Construct a Config object, overriding defaults with environment variables.

        Environment variable names are prefixed with SECURELLM_ and match field names.
        Example: SECURELLM_MODEL_QUANTIZATION_MODE=INT4
        """

        # Instantiate a temporary Config so we can access default region mapping
        default_region_map = cls().REGION_TO_DC

        return cls(
            # --- Worker Pool / Lifecycle ---
            WORKER_POOL_MIN_SIZE=_get_env_int("WORKER_POOL_MIN_SIZE", cls.WORKER_POOL_MIN_SIZE),
            WORKER_POOL_MAX_SIZE=_get_env_int("WORKER_POOL_MAX_SIZE", cls.WORKER_POOL_MAX_SIZE),
            WORKER_IDLE_TIMEOUT_MS=_get_env_int("WORKER_IDLE_TIMEOUT_MS", cls.WORKER_IDLE_TIMEOUT_MS),
            WORKER_START_RETRIES=_get_env_int("WORKER_START_RETRIES", cls.WORKER_START_RETRIES),
            WORKER_HEALTHCHECK_INTERVAL_MS=_get_env_int(
                "WORKER_HEALTHCHECK_INTERVAL_MS", cls.WORKER_HEALTHCHECK_INTERVAL_MS
            ),

            # --- Timeouts ---
            HTTP_REQUEST_TIMEOUT_MS=_get_env_int("HTTP_REQUEST_TIMEOUT_MS", cls.HTTP_REQUEST_TIMEOUT_MS),
            ORCHESTRATOR_RPC_TIMEOUT_MS=_get_env_int(
                "ORCHESTRATOR_RPC_TIMEOUT_MS", cls.ORCHESTRATOR_RPC_TIMEOUT_MS
            ),
            MODEL_LOAD_TIMEOUT_MS=_get_env_int("MODEL_LOAD_TIMEOUT_MS", cls.MODEL_LOAD_TIMEOUT_MS),
            INFERENCE_TIMEOUT_MS=_get_env_int("INFERENCE_TIMEOUT_MS", cls.INFERENCE_TIMEOUT_MS),
            STREAM_IDLE_TIMEOUT_MS=_get_env_int("STREAM_IDLE_TIMEOUT_MS", cls.STREAM_IDLE_TIMEOUT_MS),
            RAG_TIMEOUT_MS=_get_env_int("RAG_TIMEOUT_MS", cls.RAG_TIMEOUT_MS),

            # --- Model / Inference ---
            MODEL_NAME=_get_env_str("MODEL_NAME", cls.MODEL_NAME),
            MODEL_QUANTIZATION_MODE=_get_env_str("MODEL_QUANTIZATION_MODE", cls.MODEL_QUANTIZATION_MODE),
            MAX_PROMPT_TOKENS=_get_env_int("MAX_PROMPT_TOKENS", cls.MAX_PROMPT_TOKENS),
            MAX_GENERATION_TOKENS=_get_env_int("MAX_GENERATION_TOKENS", cls.MAX_GENERATION_TOKENS),
            ENABLE_FLASH_ATTENTION=_get_env_bool("ENABLE_FLASH_ATTENTION", cls.ENABLE_FLASH_ATTENTION),

            # --- RAG / Vector Retrieval ---
            RAG_ENABLED=_get_env_bool("RAG_ENABLED", cls.RAG_ENABLED),
            RAG_FALLBACK_ENABLED=_get_env_bool("RAG_FALLBACK_ENABLED", cls.RAG_FALLBACK_ENABLED),
            RAG_TOP_K=_get_env_int("RAG_TOP_K", cls.RAG_TOP_K),
            EMBEDDING_MODEL_NAME=_get_env_str("EMBEDDING_MODEL_NAME", cls.EMBEDDING_MODEL_NAME),
            VECTOR_DB_BACKEND=_get_env_str("VECTOR_DB_BACKEND", cls.VECTOR_DB_BACKEND),
            RAG_MAX_CONTEXT_TOKENS=_get_env_int("RAG_MAX_CONTEXT_TOKENS", cls.RAG_MAX_CONTEXT_TOKENS),

            # --- Security / Encryption ---
            ENCRYPTION_ALGO=_get_env_str("ENCRYPTION_ALGO", cls.ENCRYPTION_ALGO),
            KEY_ROTATION_INTERVAL_MIN=_get_env_int("KEY_ROTATION_INTERVAL_MIN", cls.KEY_ROTATION_INTERVAL_MIN),
            ALLOW_HOST_LOGGING_OF_IDS=_get_env_bool("ALLOW_HOST_LOGGING_OF_IDS", cls.ALLOW_HOST_LOGGING_OF_IDS),
            STRICT_NO_LOGGING_MODE=_get_env_bool("STRICT_NO_LOGGING_MODE", cls.STRICT_NO_LOGGING_MODE),

            # --- Metrics / Logging ---
            METRICS_ENABLED=_get_env_bool("METRICS_ENABLED", cls.METRICS_ENABLED),
            METRICS_BACKEND=_get_env_str("METRICS_BACKEND", cls.METRICS_BACKEND),
            METRICS_SAMPLING_RATE=_get_env_float("METRICS_SAMPLING_RATE", cls.METRICS_SAMPLING_RATE),
            LOG_LEVEL=_get_env_str("LOG_LEVEL", cls.LOG_LEVEL),

            # --- Multi-tenancy / Limits ---
            MAX_TENANTS=_get_env_int("MAX_TENANTS", cls.MAX_TENANTS),
            TENANT_RATE_LIMIT_QPS=_get_env_int("TENANT_RATE_LIMIT_QPS", cls.TENANT_RATE_LIMIT_QPS),
            TENANT_MAX_CONCURRENT_REQUESTS=_get_env_int(
                "TENANT_MAX_CONCURRENT_REQUESTS", cls.TENANT_MAX_CONCURRENT_REQUESTS
            ),

            # --- Region Mapping ---
            REGION_TO_DC=default_region_map.copy(),
        )



# Global config instance used throughout the codebase
config = Config.from_env()
