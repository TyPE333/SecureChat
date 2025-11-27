# src/common/config.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Self


ENV_PREFIX = "SECURELLM_"


# -------------------------------
# Environment helpers
# -------------------------------

def _env_str(name: str, default: str) -> str:
    val = os.getenv(ENV_PREFIX + name)
    return val if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(ENV_PREFIX + name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(ENV_PREFIX + name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(ENV_PREFIX + name)
    if val is None:
        return default
    v = val.lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_bytes(name: str, default: bytes) -> bytes:
    val = os.getenv(ENV_PREFIX + name)
    return val.encode("utf-8") if val else default


# -------------------------------
# Main Config Dataclass
# -------------------------------

@dataclass
class Config:
    """
    Central configuration for the SecureLLM system.
    All values can be overridden using environment variables:
    Example:
        export SECURELLM_WORKER_POOL_MIN_SIZE=2
    """

    # -------------------------
    # Network Addresses
    # -------------------------
    WORKER_ADDRESS: str = "localhost:50051"
    ORCHESTRATOR_ADDRESS: str = "localhost:60051"
    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 8000

    # -------------------------
    # Security / Encryption
    # -------------------------
    WORKER_SECRET_KEY: bytes = b"0123456789ABCDEF0123456789ABCDEF"
    ENCRYPTION_ALGO: str = "AES-GCM"
    KEY_ROTATION_INTERVAL_MIN: int = 60
    STRICT_NO_LOGGING_MODE: bool = False
    ALLOW_HOST_LOGGING_OF_IDS: bool = True

    # -------------------------
    # Worker Pool / Lifecycle
    # -------------------------
    WORKER_POOL_MIN_SIZE: int = 1
    WORKER_POOL_MAX_SIZE: int = 4
    WORKER_IDLE_TIMEOUT_MS: int = 60_000
    WORKER_START_RETRIES: int = 2
    WORKER_HEALTHCHECK_INTERVAL_MS: int = 10_000

    # -------------------------
    # Timeouts
    # -------------------------
    HTTP_REQUEST_TIMEOUT_MS: int = 30_000
    ORCHESTRATOR_RPC_TIMEOUT_MS: int = 25_000
    MODEL_LOAD_TIMEOUT_MS: int = 10_000
    INFERENCE_TIMEOUT_MS: int = 15_000
    STREAM_IDLE_TIMEOUT_MS: int = 5_000
    RAG_TIMEOUT_MS: int = 2_000

    # -------------------------
    # Model / Inference
    # -------------------------
    MODEL_NAME: str = "llm-int8"
    MODEL_QUANTIZATION_MODE: str = "INT8"  # {FP16, INT8, INT4}
    MAX_PROMPT_TOKENS: int = 2048
    MAX_GENERATION_TOKENS: int = 512
    ENABLE_FLASH_ATTENTION: bool = True

    # -------------------------
    # RAG
    # -------------------------
    RAG_ENABLED: bool = True
    RAG_FALLBACK_ENABLED: bool = True
    RAG_TOP_K: int = 4
    EMBEDDING_MODEL_NAME: str = "bge-small"
    VECTOR_DB_BACKEND: str = "faiss"
    RAG_MAX_CONTEXT_TOKENS: int = 1024

    # -------------------------
    # Multi-Tenancy Limits
    # -------------------------
    MAX_TENANTS: int = 10
    TENANT_RATE_LIMIT_QPS: int = 5
    TENANT_MAX_CONCURRENT_REQUESTS: int = 3

    # -------------------------
    # Metrics / Logging
    # -------------------------
    METRICS_ENABLED: bool = True
    METRICS_BACKEND: str = "stdout"
    METRICS_SAMPLING_RATE: float = 1.0
    LOG_LEVEL: str = "INFO"

    # -------------------------
    # Region Mapping
    # -------------------------
    REGION_TO_DC: Dict[str, str] = field(
        default_factory=lambda: {
            "us-west-1": "cluster-a",
            "us-east-1": "cluster-b",
        }
    )

    # -------------------------
    # ENV Override Loader
    # -------------------------
    @classmethod
    def from_env(cls) -> Self:
        default_region_map = cls().REGION_TO_DC

        return cls(
            # Network
            WORKER_ADDRESS=_env_str("WORKER_ADDRESS", cls.WORKER_ADDRESS),
            ORCHESTRATOR_ADDRESS=_env_str("ORCHESTRATOR_ADDRESS", cls.ORCHESTRATOR_ADDRESS),
            GATEWAY_HOST=_env_str("GATEWAY_HOST", cls.GATEWAY_HOST),
            GATEWAY_PORT=_env_int("GATEWAY_PORT", cls.GATEWAY_PORT),

            # Security
            WORKER_SECRET_KEY=_env_bytes("WORKER_SECRET_KEY", cls.WORKER_SECRET_KEY),
            ENCRYPTION_ALGO=_env_str("ENCRYPTION_ALGO", cls.ENCRYPTION_ALGO),
            KEY_ROTATION_INTERVAL_MIN=_env_int("KEY_ROTATION_INTERVAL_MIN", cls.KEY_ROTATION_INTERVAL_MIN),
            STRICT_NO_LOGGING_MODE=_env_bool("STRICT_NO_LOGGING_MODE", cls.STRICT_NO_LOGGING_MODE),
            ALLOW_HOST_LOGGING_OF_IDS=_env_bool("ALLOW_HOST_LOGGING_OF_IDS", cls.ALLOW_HOST_LOGGING_OF_IDS),

            # Worker Pool
            WORKER_POOL_MIN_SIZE=_env_int("WORKER_POOL_MIN_SIZE", cls.WORKER_POOL_MIN_SIZE),
            WORKER_POOL_MAX_SIZE=_env_int("WORKER_POOL_MAX_SIZE", cls.WORKER_POOL_MAX_SIZE),
            WORKER_IDLE_TIMEOUT_MS=_env_int("WORKER_IDLE_TIMEOUT_MS", cls.WORKER_IDLE_TIMEOUT_MS),
            WORKER_START_RETRIES=_env_int("WORKER_START_RETRIES", cls.WORKER_START_RETRIES),
            WORKER_HEALTHCHECK_INTERVAL_MS=_env_int(
                "WORKER_HEALTHCHECK_INTERVAL_MS", cls.WORKER_HEALTHCHECK_INTERVAL_MS
            ),

            # Timeouts
            HTTP_REQUEST_TIMEOUT_MS=_env_int("HTTP_REQUEST_TIMEOUT_MS", cls.HTTP_REQUEST_TIMEOUT_MS),
            ORCHESTRATOR_RPC_TIMEOUT_MS=_env_int("ORCHESTRATOR_RPC_TIMEOUT_MS", cls.ORCHESTRATOR_RPC_TIMEOUT_MS),
            MODEL_LOAD_TIMEOUT_MS=_env_int("MODEL_LOAD_TIMEOUT_MS", cls.MODEL_LOAD_TIMEOUT_MS),
            INFERENCE_TIMEOUT_MS=_env_int("INFERENCE_TIMEOUT_MS", cls.INFERENCE_TIMEOUT_MS),
            STREAM_IDLE_TIMEOUT_MS=_env_int("STREAM_IDLE_TIMEOUT_MS", cls.STREAM_IDLE_TIMEOUT_MS),
            RAG_TIMEOUT_MS=_env_int("RAG_TIMEOUT_MS", cls.RAG_TIMEOUT_MS),

            # Model
            MODEL_NAME=_env_str("MODEL_NAME", cls.MODEL_NAME),
            MODEL_QUANTIZATION_MODE=_env_str("MODEL_QUANTIZATION_MODE", cls.MODEL_QUANTIZATION_MODE),
            MAX_PROMPT_TOKENS=_env_int("MAX_PROMPT_TOKENS", cls.MAX_PROMPT_TOKENS),
            MAX_GENERATION_TOKENS=_env_int("MAX_GENERATION_TOKENS", cls.MAX_GENERATION_TOKENS),
            ENABLE_FLASH_ATTENTION=_env_bool("ENABLE_FLASH_ATTENTION", cls.ENABLE_FLASH_ATTENTION),

            # RAG
            RAG_ENABLED=_env_bool("RAG_ENABLED", cls.RAG_ENABLED),
            RAG_FALLBACK_ENABLED=_env_bool("RAG_FALLBACK_ENABLED", cls.RAG_FALLBACK_ENABLED),
            RAG_TOP_K=_env_int("RAG_TOP_K", cls.RAG_TOP_K),
            EMBEDDING_MODEL_NAME=_env_str("EMBEDDING_MODEL_NAME", cls.EMBEDDING_MODEL_NAME),
            VECTOR_DB_BACKEND=_env_str("VECTOR_DB_BACKEND", cls.VECTOR_DB_BACKEND),
            RAG_MAX_CONTEXT_TOKENS=_env_int("RAG_MAX_CONTEXT_TOKENS", cls.RAG_MAX_CONTEXT_TOKENS),

            # Multi-Tenancy
            MAX_TENANTS=_env_int("MAX_TENANTS", cls.MAX_TENANTS),
            TENANT_RATE_LIMIT_QPS=_env_int("TENANT_RATE_LIMIT_QPS", cls.TENANT_RATE_LIMIT_QPS),
            TENANT_MAX_CONCURRENT_REQUESTS=_env_int(
                "TENANT_MAX_CONCURRENT_REQUESTS", cls.TENANT_MAX_CONCURRENT_REQUESTS
            ),

            # Metrics
            METRICS_ENABLED=_env_bool("METRICS_ENABLED", cls.METRICS_ENABLED),
            METRICS_BACKEND=_env_str("METRICS_BACKEND", cls.METRICS_BACKEND),
            METRICS_SAMPLING_RATE=_env_float("METRICS_SAMPLING_RATE", cls.METRICS_SAMPLING_RATE),
            LOG_LEVEL=_env_str("LOG_LEVEL", cls.LOG_LEVEL),

            # Region Mapping
            REGION_TO_DC=default_region_map.copy(),
        )


# Global configuration (imported everywhere)
config = Config.from_env()
