# src/gateway/orchestrator_client.py

import grpc
import orchestrator_pb2
import orchestrator_pb2_grpc

from common.logging_utils import log_event


class OrchestratorClient:
    def __init__(self, address: str):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = orchestrator_pb2_grpc.OrchestratorServiceStub(self.channel)

    async def dispatch_inference(self, request_id: str, http_request):
        """
        Convert HTTP request → gRPC request,
        send to orchestrator, and return the streaming generator.
        """

        # For MVP, we treat prompt as plaintext → orchestrator encrypts it
        grpc_request = orchestrator_pb2.InferenceRequest(
            tenant_id=http_request.tenant_id,
            region=http_request.region,
            mode=http_request.mode,
            client_pubkey=http_request.client_pubkey or "",
            request_id=request_id,
            encrypted_prompt=http_request.prompt.encode("utf-8")
        )

        log_event("gateway_calling_orchestrator", request_id=request_id)

        # Streaming call
        response_stream = self.stub.DispatchInference(grpc_request)

        return response_stream
