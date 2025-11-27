# src/orchestrator/inference_dispatcher.py

import uuid
from common.encryption import encrypt_blob
from common.logging_utils import log_event
from common.config import config


class InferenceDispatcher:
    """
    Handles:
    - request ID generation
    - encrypting prompt
    - sending to worker client
    """

    def __init__(self, worker_client, worker_key: bytes):
        self.worker_client = worker_client
        self.worker_key = worker_key

    def dispatch(self, request):
        """
        request is InferenceRequest (proto)
        """

        # Use request_id provided or generate new
        request_id = request.request_id or str(uuid.uuid4())

        log_event("orchestrator_received_request", request_id=request_id)

        # request.encrypted_prompt is already encrypted by the Gateway (later)
        # For now, assume prompt is plaintext for MVP
        prompt_bytes = request.encrypted_prompt  # MVP: treat as cleartext for now

        # Encrypt for worker using worker_key (simulated enclave boundary)
        encrypted_blob = encrypt_blob(prompt_bytes, self.worker_key)

        log_event("orchestrator_encrypted_payload", request_id=request_id)

        # Call worker
        return request_id, encrypted_blob
