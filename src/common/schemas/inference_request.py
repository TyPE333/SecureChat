from dataclasses import dataclass
from typing import Optional

@dataclass
class InferenceRequest:
    """
    Internal python representation of a client inference request.
    This object is created by Gateway after HTTP validation
    and before sending gRPC call to the orchestrator.
    """

    tenant_id: str
    prompt: str
    mode: str    #plain or rag
    region: str
    client_pubkey: Optional[str] = None
    request_id: Optional[str] = None
