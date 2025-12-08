# src/workers/inference_engine.py

import threading

import torch
from transformers import TextIteratorStreamer

from common.config import config


class InferenceEngine:
    """
    Wraps a GPU model and exposes streaming helpers on top of HF `generate`.

    Flow (streaming):
      - Spin up a background `model.generate(..., streamer=...)` call.
      - Yield decoded chunks from `TextIteratorStreamer` as soon as HF emits
        them so the worker/orchestrator pipeline can forward bytes immediately.

    Metrics:
      - last_input_tokens: number of prompt tokens for the last request
      - last_output_tokens: approximate number of generated “tokens” (chunks)
        for the last request (good enough for perf reasoning).
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

        # Detect 8-bit / 4-bit quantized models (bitsandbytes)
        is_8bit = bool(getattr(self.model, "is_loaded_in_8bit", False))
        is_4bit = bool(getattr(self.model, "is_loaded_in_4bit", False))

        if is_8bit or is_4bit:
            # HF already placed the model on the correct device; do NOT call .to()
            # We just introspect one parameter to know which device to send inputs to.
            try:
                param = next(self.model.parameters())
                self.device = param.device
            except StopIteration:
                # Extremely unlikely, but be defensive
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Non-quantized model: we control the device placement
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        self.model.eval()

        # Some models don't have pad_token_id set properly; be defensive.
        if getattr(self.model.config, "pad_token_id", None) is None:
            if getattr(self.model.config, "eos_token_id", None) is not None:
                self.model.config.pad_token_id = self.model.config.eos_token_id

        # ---- Metrics fields ----
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    # -------------------------------
    # Internal helpers
    # -------------------------------
    def _encode(self, prompt: str):
        """
        Tokenize and encode prompt to device tensor.

        Also tracks input token length for metrics via last_input_tokens.
        """
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_PROMPT_TOKENS,
        )

        # shape: (batch_size=1, seq_len)
        self.last_input_tokens = enc["input_ids"].shape[1]

        return enc.to(self.device)

    def _build_generation_kwargs(self, prompt: str) -> dict:
        """Pack shared generation kwargs so streaming and blocking paths match."""
        inputs = self._encode(prompt)

        return dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=config.MAX_GENERATION_TOKENS,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            # Important: we explicitly disable KV cache here because the Qwen2 2B
            # INT8 setup previously hit an internal index error with cache logic.
            use_cache=False,
        )

    def _generate_full_text(self, prompt: str) -> str:
        """
        Blocking, full-text generation (non-streaming) using HF generate.

        We explicitly disable KV cache here (`use_cache=False`) because some
        model/quantization combos (like our Qwen2 2B INT8) can hit an internal
        index error in the cache logic.
        """
        gen_kwargs = self._build_generation_kwargs(prompt)

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        # For metrics: how many tokens did we generate?
        # (This is approximate: total_len - input_len)
        try:
            total_len = output_ids.shape[1]
            self.last_output_tokens = max(total_len - self.last_input_tokens, 0)
        except Exception:
            self.last_output_tokens = 0

        text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        return text

    # -------------------------------
    # Public API used by worker_server
    # -------------------------------
    def generate_stream(self, prompt: str):
        """
        Generator that yields chunks of text as "tokens" to the worker.

        Implementation detail:
          - Launch HF generation with `TextIteratorStreamer` on a background
            thread so we don't block the caller.
          - Yield decoded chunks immediately as the streamer produces them.

        Note:
          - We approximate output token count by counting the number of
            chunks produced by the streamer. This is good enough for
            latency/throughput reasoning in this project.
        """
        # Reset output count for this request
        self.last_output_tokens = 0

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            decode_with_prefix_space=True,
        )

        gen_kwargs = self._build_generation_kwargs(prompt)
        gen_kwargs["streamer"] = streamer

        def _run_generation():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        threading.Thread(target=_run_generation, daemon=True).start()

        for text in streamer:
            if text:
                self.last_output_tokens += 1
                yield text

    def run(self, prompt: str):
        """
        Backward-compatible entrypoint used by `WorkerService.RunInference`.
        """
        return self.generate_stream(prompt)

    def stream_generate(self, prompt: str):
        """Alias kept for compatibility if referenced elsewhere."""
        return self.generate_stream(prompt)

    def generate_full(self, prompt: str) -> str:
        """Convenience method if you ever want full text in one shot."""
        return self._generate_full_text(prompt)