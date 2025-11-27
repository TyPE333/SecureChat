# src/gateway/request_models.py

from pydantic import BaseModel, Field


class InferenceHTTPRequest(BaseModel):
    tenant_id: str = Field(..., example="tenant-1")
    prompt: str = Field(..., example="Hello, world")
    mode: str = Field(..., pattern="^(plain|rag)$")
    region: str = Field(..., example="us-west-1")
    client_pubkey: str = Field(None, example="optional-public-key")
