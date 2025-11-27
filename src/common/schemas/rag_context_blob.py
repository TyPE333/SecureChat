from dataclasses import dataclass

@dataclass
class RAGContextBlob:
    """
    Output of the RAG subsystem. This enriched prompt/context
    is encrypted by the Host (Gateway/Orchestrator) before
    being packaged into EncryptedPayload.
    """
    rag_context: str 