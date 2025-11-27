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

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

loop_thread = futures.ThreadPoolExecutor(max_workers=1)
loop_thread.submit(start_loop, ORCH_LOOP)


class OrchestratorService(orchestrator_pb2_grpc.OrchestratorServiceServicer):

    def __init__(self):
        self.worker_client = WorkerClient(config.WORKER_ADDRESS)
        self.dispatcher = InferenceDispatcher(
            worker_client=self.worker_client,
            worker_key=config.WORKER_SECRET_KEY
        )

    def DispatchInference(self, request, context):
        """
        SYNCHRONOUS gRPC handler (required by grpcio).
        But internally launches async worker-stream via ORCH_LOOP.
        """

        # Prepare encrypted payload
        request_id, encrypted_blob = self.dispatcher.dispatch(request)
        log_event("orchestrator_received_request", request_id=request_id)

        queue: asyncio.Queue[bytes] = asyncio.Queue()

        async def async_worker_task():
            """Async coroutine executed in ORCH_LOOP."""
            try:
                log_event("orchestrator_calling_worker", request_id=request_id)

                # Blocking worker RPC (safe in async task)
                worker_stream = self.worker_client.run_inference(
                    request_id=request_id,
                    encrypted_blob=encrypted_blob
                )

                log_event("orchestrator_streaming_tokens", request_id=request_id)

                for token_msg in worker_stream:
                    token = token_msg.encrypted_token
                    frame = struct.pack(">I", len(token)) + token
                    await queue.put(frame)

            finally:
                await queue.put(None)

        # Schedule the async task safely in ORCH_LOOP
        asyncio.run_coroutine_threadsafe(async_worker_task(), ORCH_LOOP)

        # SYNC GENERATOR: yield frames produced by async task
        while True:
            frame_future = asyncio.run_coroutine_threadsafe(queue.get(), ORCH_LOOP)
            frame = frame_future.result()

            if frame is None:
                break

            yield orchestrator_pb2.EncryptedToken(encrypted_token=frame)

        log_event("orchestrator_request_completed", request_id=request_id)


def serve(port=60051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorService(), server
    )

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    log_event("orchestrator_started", extra={"port": str(port)})

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
