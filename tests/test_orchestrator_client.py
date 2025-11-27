# src/tests/test_orchestrator_client.py

import grpc
import orchestrator_pb2
import orchestrator_pb2_grpc

def main():
    channel = grpc.insecure_channel("localhost:60051")
    stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)

    # For MVP, we treat encrypted_prompt as plaintext
    request = orchestrator_pb2.InferenceRequest(
        tenant_id="test-tenant",
        region="us-west-1",
        mode="plain",
        client_pubkey="none",
        request_id="req-123",
        encrypted_prompt=b"Hello world (plaintext for MVP)"
    )

    print("Sending request...")

    response_stream = stub.DispatchInference(request)

    print("Streaming response from orchestrator:\n")

    for token in response_stream:
        print("Encrypted token chunk:", token.encrypted_token)


if __name__ == "__main__":
    main()
