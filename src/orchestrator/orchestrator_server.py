# src/orchestrator/orchestrator_server.py

import grpc
import struct
import asyncio
from concurrent import futures

import orchestrator_pb2
import orchestrator_pb2_grpc

from orchestrator.inference_dispatcher import InferenceDispatcher
from orchestrator.worker_client import WorkerClient

from common.config import config
from common.logging_utils import log_event

# Single shared orchestrator event loop in a background thread
ORCH_LOOP = asyncio.new_event_loop()


def _start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


loop_thread = futures.ThreadPoolExecutor(max_workers=1)
loop_thread.submit(_start_loop, ORCH_LOOP)


class OrchestratorService(orchestrator_pb2_grpc.OrchestratorServiceServicer):
    """
    Orchestrator sits between the HTTP gateway and one or more worker processes.

    This version supports a simple multi-worker setup via round-robin:
      - We maintain a list of (worker_id, WorkerClient) entries.
      - Each incoming request picks the next worker in the list.
    """

    def __init__(self):
        # --- Worker clients (multi-worker aware) ---
        #
        # For now we hard-code two workers on different ports.
        # You can tweak these addresses or read from env later if you want.
        self.workers = [
            ("worker-1", WorkerClient("localhost:50051")),
            ("worker-2", WorkerClient("localhost:50052")),
        ]

        if not self.workers:
            raise RuntimeError("Orchestrator has no workers configured")

        # Round-robin index
        self._rr_index = 0

        # Dispatcher only cares about encrypting payloads now.
        # We keep the API signature but pass None for worker_client.
        self.dispatcher = InferenceDispatcher(
            worker_client=None,
            worker_key=config.WORKER_SECRET_KEY,
        )

    # -------------------------
    # Worker selection
    # -------------------------
    def _choose_worker(self):
        """
        Simple round-robin over the available workers.
        Returns (worker_id, WorkerClient).
        """
        idx = self._rr_index % len(self.workers)
        self._rr_index = (self._rr_index + 1) % len(self.workers)
        return self.workers[idx]

    # -------------------------
    # gRPC: DispatchInference
    # -------------------------
    def DispatchInference(self, request, context):
        """
        SYNCHRONOUS gRPC handler (required by grpcio).
        Internally launches async worker-stream via ORCH_LOOP.
        """

        # Prepare encrypted payload (and get request_id)
        request_id, encrypted_blob = self.dispatcher.dispatch(request)

        log_event(
            "orchestrator_received_request",
            request_id=request_id,
            extra={"strict_no_logging": str(config.STRICT_NO_LOGGING_MODE)},
        )

        # Choose a worker for this request (round-robin)
        worker_id, worker_client = self._choose_worker()

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def async_worker_task():
            """Async coroutine executed in ORCH_LOOP."""
            try:
                log_event(
                    "orchestrator_calling_worker",
                    request_id=request_id,
                    extra={
                        "worker_id": worker_id,
                        "worker_address": worker_client.address,
                    },
                )

                # Blocking worker RPC (safe in this background task)
                try:
                    worker_stream = worker_client.run_inference(
                        request_id=request_id,
                        encrypted_blob=encrypted_blob,
                    )
                except Exception as e:
                    log_event(
                        "orchestrator_worker_rpc_error",
                        request_id=request_id,
                        error=repr(e),
                        level="error",
                    )
                    return  # queue cleanup happens in finally

                log_event(
                    "orchestrator_streaming_tokens",
                    request_id=request_id,
                    extra={"worker_id": worker_id},
                )

                for token_msg in worker_stream:
                    token = token_msg.encrypted_token
                    frame = struct.pack(">I", len(token)) + token
                    await queue.put(frame)

            finally:
                # Signal end of stream to the sync side
                await queue.put(None)

        # Schedule the async task in ORCH_LOOP
        asyncio.run_coroutine_threadsafe(async_worker_task(), ORCH_LOOP)

        # SYNC GENERATOR: yield frames produced by async task
        while True:
            frame_future = asyncio.run_coroutine_threadsafe(queue.get(), ORCH_LOOP)
            frame = frame_future.result()

            if frame is None:
                break

            yield orchestrator_pb2.EncryptedToken(encrypted_token=frame)

        log_event(
            "orchestrator_request_completed",
            request_id=request_id,
            extra={"strict_no_logging": str(config.STRICT_NO_LOGGING_MODE)},
        )


def serve(port: int = 60051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorService(), server
    )

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    log_event(
        "orchestrator_started",
        extra={
            "port": str(port),
            "strict_no_logging": str(config.STRICT_NO_LOGGING_MODE),
        },
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()