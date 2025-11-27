from dataclasses import dataclass

@dataclass
class EncryptedPayload:
    """
    Internal representation of the encrypted blob that is sent
    from the or chestrator to the worker via gRPC.
    """
    request_id: str
    encrypted_blob: bytes  #Contains encrypted prompt and metadata