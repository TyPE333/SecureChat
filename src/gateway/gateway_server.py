# src/gateway/gateway_server.py

import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from gateway.orchestrator_client import OrchestratorClient
from gateway.request_models import InferenceHTTPRequest

from common.logging_utils import log_event

# Create FastAPI app
app = FastAPI(title="SecureChat Gateway MVP")

# Orchestrator is running at port 60051
orchestrator = OrchestratorClient("localhost:60051")


# --------------------------------------------------------
# /infer endpoint
# --------------------------------------------------------
@app.post("/infer")
async def infer(request: InferenceHTTPRequest):

    request_id = str(uuid.uuid4())

    log_event(
        "gateway_received_request",
        request_id=request_id,
        tenant_id=request.tenant_id,
        extra={"mode": request.mode}
    )

    # Make async call to orchestrator (streaming gRPC)
    response_stream = await orchestrator.dispatch_inference(
        request_id=request_id,
        http_request=request
    )

    async def stream_tokens():
        """
        Convert gRPC streaming into HTTP streaming.
        The tokens are encrypted bytes; we simply forward them.
        """
        for token_msg in response_stream:
            chunk = token_msg.encrypted_token + b"\n"
            yield chunk

        log_event(
            "gateway_streaming_completed",
            request_id=request_id,
            tenant_id=request.tenant_id
        )

    return StreamingResponse(stream_tokens(), media_type="application/octet-stream")


# --------------------------------------------------------
# Entry point for running via: python src/gateway/gateway_server.py
# --------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
