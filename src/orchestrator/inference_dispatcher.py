# src/orchestrator/inference_dispatcher.py

from __future__ import annotations

import uuid
from typing import Tuple

from common.encryption import hybrid_encrypt_for_worker, get_worker_public_key
from common.logging_utils import log_event
from common.config import config


class InferenceDispatcher:
    """
    Orchestrator-side helper that:
    - Normalizes/assigns request_id.
    - Prepares the plaintext payload for the worker (currently just the prompt).
    - Encrypts that payload using hybrid RSA+AES for the worker enclave.
    """

    def __init__(self, worker_client, worker_key: bytes):
        # We keep worker_client and worker_key parameters for API compatibility,
        # but the encryption is now purely done via hybrid RSA+AES.
        self.worker_client = worker_client
        self.worker_key = worker_key  # no longer used for encryption

    def _ensure_request_id(self, request) -> str:
        """
        Returns an existing request_id if present, otherwise generates a new one.
        """
        req_id = getattr(request, "request_id", None)
        if not req_id:
            req_id = str(uuid.uuid4())
        return req_id

    def dispatch(self, request) -> Tuple[str, bytes]:
        """
        Prepare an encrypted payload for the worker.

        Returns
        -------
        request_id : str
            Unique identifier for this inference request.
        encrypted_blob : bytes
            Opaque hybrid-encrypted blob to send to the worker.
        """
        # Normalize / generate request_id
        request_id = self._ensure_request_id(request)

        # For now, the worker only needs the prompt text as plaintext.
        # Gateway currently stuffs the prompt bytes into `encrypted_prompt`
        # (MVP placeholder), so fall back to request.prompt if present.
        gateway_payload = getattr(request, "encrypted_prompt", None)
        if gateway_payload:
            plaintext = bytes(gateway_payload)
        else:
            prompt = getattr(request, "prompt", "")
            plaintext = prompt.encode("utf-8")

        # Hybrid RSA+AES encryption:
        # - Orchestrator uses the worker enclave's *public* key.
        # - Worker will decrypt with its private key.
        worker_pub = get_worker_public_key()
        encrypted_blob = hybrid_encrypt_for_worker(plaintext, worker_pub)

        # Optional: log that we prepared the encrypted payload (no content)
        log_event(
            "orchestrator_encrypted_payload",
            request_id=request_id,
            extra={"strict_no_logging": str(config.STRICT_NO_LOGGING_MODE)},
        )

        return request_id, encrypted_blob
