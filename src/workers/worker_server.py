# src/workers/worker_server.py

import grpc
from concurrent import futures

from worker_pb2 import EncryptedToken
import worker_pb2_grpc

from common.encryption import decrypt_blob, encrypt_token
from common.logging_utils import log_event
from common.schemas.worker_state import WorkerState
from common.config import config

from workers.model_loader import GPUModelLoader
from workers.inference_engine import InferenceEngine
from workers.worker_state_manager import WorkerStateManager


class WorkerService(worker_pb2_grpc.WorkerServiceServicer):
    def __init__(self, worker_id: str):
        self.worker_id = worker_id

        # Shared AES key for MVP (Orchestrator & Worker use the SAME key)
        self.worker_key = config.WORKER_SECRET_KEY

        self.state_manager = WorkerStateManager(worker_id)
        
        # Load model + tokenizer using GPU loader
        self.model_loader = GPUModelLoader(worker_id)
        self.engine = None

        self.initialize_worker()

    # --------------------------------------------------
    # Worker initialization
    # --------------------------------------------------
    def initialize_worker(self):
        self.state_manager.set_state(WorkerState.MODEL_LOADING)
        self.model_loader.load()

        self.engine = InferenceEngine(
            tokenizer=self.model_loader.tokenizer,
            model=self.model_loader.model,
        )

        self.state_manager.set_state(WorkerState.READY)

    # --------------------------------------------------
    # gRPC: RunInference
    # --------------------------------------------------
    def RunInference(self, request, context):
        request_id = request.request_id

        self.state_manager.set_state(WorkerState.BUSY, request_id=request_id)

        # Decrypt incoming encrypted blob
        try:
            plaintext = decrypt_blob(request.encrypted_blob, self.worker_key)
        except Exception:
            self.state_manager.set_state(WorkerState.FAILED, request_id=request_id)
            log_event(
                "decryption_failed",
                worker_id=self.worker_id,
                request_id=request_id,
                level="error"
            )
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid encrypted payload")
            return

        prompt = plaintext.decode("utf-8")
        log_event(
            "inference_s tarted",
            worker_id=self.worker_id,
            request_id=request_id,
        )

        self.state_manager.set_state(WorkerState.STREAMING, request_id=request_id)

        # Streaming inference using Qwen2.5 Instruct
        for token in self.engine.stream_generate(prompt):
            if not token.strip():
                continue # Skip empty tokens

            encrypted = encrypt_token(token, self.worker_key)
            yield EncryptedToken(encrypted_token=encrypted)

        log_event(
            "inference_completed",
            worker_id=self.worker_id,
            request_id=request_id,
        )

        self.state_manager.set_state(WorkerState.READY, request_id=request_id)


# ------------------------------------------------------
# Worker server
# ------------------------------------------------------
def serve(worker_id: str = "worker-1", port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(
        WorkerService(worker_id), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    log_event(
        "worker_server_started",
        worker_id=worker_id,
        extra={"port": str(port)}
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
