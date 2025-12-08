# src/gateway/gateway_server.py

import uuid
import time
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from gateway.orchestrator_client import OrchestratorClient
from gateway.request_models import InferenceHTTPRequest
from common.logging_utils import log_event
from common.config import config


app = FastAPI(title="SecureLLM Gateway MVP")

orchestrator = OrchestratorClient(config.ORCHESTRATOR_ADDRESS)


@app.post("/infer")
async def infer(request: InferenceHTTPRequest):
    request_id = str(uuid.uuid4())

    # Mark start of end-to-end timing at the HTTP boundary
    start_time = time.monotonic()

    log_event(
        "gateway_received_request",
        request_id=request_id,
        tenant_id=request.tenant_id,
        extra={"mode": request.mode},
    )

    # Blocking gRPC stream call → will be run in executor
    response_stream = await orchestrator.dispatch_inference(
        request_id=request_id,
        http_request=request,
    )

    # Queue for async streaming
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    # Simple counters for streaming metrics
    total_bytes = 0
    frame_count = 0

    def stream_worker():
        """Run in a background thread, pulling from gRPC stream."""
        try:
            for token_msg in response_stream:  # synchronous iterator
                queue.put_nowait(token_msg.encrypted_token)
        except Exception as e:
            print("Gateway stream_worker error:", e)
        finally:
            # Signal end of stream
            queue.put_nowait(None)

    # Launch worker in thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, stream_worker)

    # Async streaming generator that FastAPI will use
    async def async_streamer():
        nonlocal total_bytes, frame_count

        while True:
            chunk = await queue.get()
            if chunk is None:
                break

            # Update simple streaming metrics
            frame_count += 1
            total_bytes += len(chunk)

            yield chunk

        # End of stream — log completion + end-to-end timing
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        log_event(
            "gateway_streaming_completed",
            request_id=request_id,
            tenant_id=request.tenant_id,
        )

        log_event(
            "gateway_end_to_end_metrics",
            request_id=request_id,
            tenant_id=request.tenant_id,
            extra={
                "total_ms": str(elapsed_ms),
                "frames": str(frame_count),
                "bytes": str(total_bytes),
            },
        )

    return StreamingResponse(async_streamer(), media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.GATEWAY_HOST, port=config.GATEWAY_PORT)