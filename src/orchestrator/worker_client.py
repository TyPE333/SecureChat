# src/orchestrator/worker_client.py

import grpc

from worker_pb2 import EncryptedPayload
import worker_pb2_grpc

from common.logging_utils import log_event


class WorkerClient:
    """
    Very simple Worker gRPC client for MVP.
    Assumes a single worker running at a fixed address.
    """

    def __init__(self, address: str):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = worker_pb2_grpc.WorkerServiceStub(self.channel)

    def run_inference(self, request_id: str, encrypted_blob: bytes):
        """
        Call the worker with an encrypted payload.
        Return a generator over encrypted tokens.
        """
        log_event(
            "worker_client_run_inference",
            request_id=request_id,
            extra={
                "worker_address": self.address,
                "blob_len": str(len(encrypted_blob)),
            },
        )
        payload = EncryptedPayload(
            request_id=request_id,
            encrypted_blob=encrypted_blob
        )

        log_event("orchestrator_calling_worker", request_id=request_id)

        return self.stub.RunInference(payload)
