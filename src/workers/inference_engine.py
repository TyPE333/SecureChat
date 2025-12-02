# src/workers/inference_engine.py

import torch
import logging
from transformers import TextIteratorStreamer

from common.config import config

class InferenceEngine:
    """
    Wraps a GPU model and provides streaming token generation
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

        self.model.eval()
        self.device = "cuda:0"
    
    def _encode(self, prompt: str):
        "Tokenize and encode prompt to tensor"
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_PROMPT_TOKENS,
        ).to(self.device)
    
    def generate_stream(self, prompt: str):
        """
        Stream tokens one-by-one using HF TextIteratorStreamer.
        Yields strings (decoded tokens).
        """

        inputs = self._encode(prompt)

        # Create the streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation config
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=config.MAX_GENERATION_TOKENS,
            do_sample=False,
            temperature=0.0,
            streamer=streamer,
        )

        # Launch generation in background thread
        import threading
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield tokens as they become available
        for token in streamer:
            yield token

    def stream_generate(self, prompt: str):
        """Backward compatible alias for generate_stream."""
        return self.generate_stream(prompt)
    
    def generate_full(self, prompt: str) -> str:
        """ Synchronous full generation (non-streaming) """
        inputs = self._encode(prompt)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=config.MAX_GENERATION_TOKENS,
            do_sample=False,
            temperature=0.0,
        )

        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return output_text
