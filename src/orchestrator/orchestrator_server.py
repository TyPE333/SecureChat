# src/orchestrator/orchestrator_server.py

import grpc
from concurrent import futures

import orchestrator_pb2
import orchestrator_pb2_grpc

from orchestrator.inference_dispatcher import InferenceDispatcher
from orchestrator.worker_client import WorkerClient

from common.config import config
from common.logging_utils import log_event


class OrchestratorService(orchestrator_pb2_grpc.OrchestratorServiceServicer):
    def __init__(self):
        # MVP: single worker at a fixed address
        self.worker_address = "localhost:50051"
        self.worker_client = WorkerClient(self.worker_address)

        # Shared AES key (same key used by worker)
        self.worker_key = config.WORKER_SECRET_KEY

        self.dispatcher = InferenceDispatcher(
            worker_client=self.worker_client,
            worker_key=self.worker_key
        )

    def DispatchInference(self, request, context):
        """
        MVP:
        - request.encrypted_prompt is treated as plaintext bytes
          (we will fix this when we build the Gateway)
        """
        request_id, encrypted_blob = self.dispatcher.dispatch(request)

        worker_stream = self.worker_client.run_inference(
            request_id=request_id,
            encrypted_blob=encrypted_blob
        )

        log_event("orchestrator_streaming_tokens", request_id=request_id)

        # Stream tokens back to caller
        for token_msg in worker_stream:
            yield orchestrator_pb2.EncryptedToken(
                encrypted_token=token_msg.encrypted_token
            )

        log_event("orchestrator_request_completed", request_id=request_id)


# ------------------------------------------------------
# Orchestrator server
# ------------------------------------------------------
def serve(port: int = 60051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorService(), server
    )

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    log_event("orchestrator_started", extra={"port": str(port)})

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
