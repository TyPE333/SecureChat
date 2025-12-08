# src/workers/worker_server.py
import argparse
import grpc
from concurrent import futures
from time import perf_counter

from worker_pb2 import EncryptedToken
import worker_pb2_grpc

from common.encryption import (
    hybrid_decrypt_at_worker,
    encrypt_token,
    get_worker_private_key,
)
from common.logging_utils import log_event
from common.schemas.worker_state import WorkerState
from common.config import config

from workers.model_loader import GPUModelLoader
from workers.inference_engine import InferenceEngine
from workers.worker_state_manager import WorkerStateManager


class WorkerService(worker_pb2_grpc.WorkerServiceServicer):
    def __init__(self, worker_id: str):
        self.worker_id = worker_id

        # This key is now only used for *response* encryption (worker -> host).
        # The host treats the bytes as opaque; it does not decrypt them.
        self.worker_key = config.WORKER_SECRET_KEY

        self.state_manager = WorkerStateManager(worker_id)
        self.model_loader = GPUModelLoader(worker_id)
        self.engine = None

        self.initialize_worker()

    # --------------------------------------------------
    # Worker initialization
    # --------------------------------------------------
    def initialize_worker(self):
        self.state_manager.set_state(WorkerState.MODEL_LOADING)
        tokenizer, model = self.model_loader.load()
        self.engine = InferenceEngine(tokenizer, model)
        self.state_manager.set_state(WorkerState.READY)

    # --------------------------------------------------
    # gRPC: RunInference
    # --------------------------------------------------
    def RunInference(self, request, context):
        request_id = request.request_id

        # Prove the RPC hit the worker
        log_event(
            "worker_run_inference_called",
            worker_id=self.worker_id,
            request_id=request_id,
        )

        self.state_manager.set_state(WorkerState.BUSY, request_id=request_id)

        # Hybrid decrypt: unwrap RSA+AES envelope sent by orchestrator
        try:
            plaintext = hybrid_decrypt_at_worker(
                request.encrypted_blob,
                get_worker_private_key(),
            )
        except Exception as e:
            self.state_manager.set_state(WorkerState.FAILED, request_id=request_id)
            log_event(
                "decryption_failed",
                worker_id=self.worker_id,
                request_id=request_id,
                error=repr(e),
                level="error",
            )
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid encrypted payload")
            return

        prompt = plaintext.decode("utf-8")

        # ---- Metrics: start timing here ----
        t_start = perf_counter()
        t_first_token = None
        token_count = 0

        log_event(
            "inference_started",
            worker_id=self.worker_id,
            request_id=request_id,
        )

        self.state_manager.set_state(WorkerState.STREAMING, request_id=request_id)

        try:
            # Stream encrypted tokens back to orchestrator.
            # Note: orchestrator/gateway do NOT decrypt these; they remain opaque.
            for token in self.engine.run(prompt):
                if t_first_token is None:
                    t_first_token = perf_counter()
                token_count += 1

                encrypted = encrypt_token(token, self.worker_key)
                yield EncryptedToken(encrypted_token=encrypted)

            # End timing once generation finishes
            t_end = perf_counter()

            # Compute metrics
            if t_first_token is None:
                # No tokens produced; treat TTFT as total latency
                t_first_token = t_end

            ttft_ms = int((t_first_token - t_start) * 1000)
            total_ms = int((t_end - t_start) * 1000)
            tokens_per_sec = (
                float(token_count) / (total_ms / 1000.0)
                if token_count > 0 and total_ms > 0
                else 0.0
            )

            # Log metrics (no content, only metadata)
            input_tokens = getattr(self.engine, "last_input_tokens", 0)
            output_tokens = getattr(self.engine, "last_output_tokens", token_count)

            log_event(
                "inference_metrics",
                worker_id=self.worker_id,
                request_id=request_id,
                extra={
                    "ttft_ms": str(ttft_ms),
                    "total_ms": str(total_ms),
                    "tokens_per_sec": f"{tokens_per_sec:.2f}",
                    "input_tokens": str(input_tokens),
                    "output_tokens": str(output_tokens),
                },
            )

            log_event(
                "inference_completed",
                worker_id=self.worker_id,
                request_id=request_id,
            )

        except Exception as e:
            # If generation fails mid-stream, record failure
            log_event(
                "inference_error",
                worker_id=self.worker_id,
                request_id=request_id,
                error=repr(e),
                level="error",
            )
            self.state_manager.set_state(WorkerState.FAILED, request_id=request_id)
            context.abort(grpc.StatusCode.INTERNAL, "Worker inference failed")
        finally:
            # Unless we aborted early, return to READY
            # (Calling READY even after FAILED is harmless for this demo.)
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
        extra={"port": str(port)},
    )

    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=str, default="worker-1", help="Unique worker ID")
    parser.add_argument("--port", type=int, default=50051, help="gRPC")
    args = parser.parse_args()

    serve(worker_id=args.worker_id, port=args.port)
                        
