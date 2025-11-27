# src/workers/inference_engine.py

from typing import Generator, List

class InferenceEngine:
    """
    Mock inference: yields fixed tokens.
    Replace this with real model inference later.
    """

    def __init__(self):
        pass

    def run(self, prompt: str) -> Generator[str, None, None]:
        mock_tokens: List[str] = [
            "This", "is", "a", "mock", "response", "from", "the", "worker."
        ]
        for token in mock_tokens:
            yield token
