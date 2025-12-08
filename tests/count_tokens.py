from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

short_prompt = "Explain what Transformer models are and how self-attention works. Keep it to a few short paragraphs."

medium_prompt = """You are an expert ML systems engineer working on a privacy-preserving LLM inference service. The system has a gateway, an orchestrator, and a GPU worker pool hosting Qwen2.5-1.5B-Instruct in FP16. Each request is encrypted end-to-end, and we care a lot about latency, throughput, and tail behavior under load.

Given this context, explain in detail what Transformer models are and how they work. Cover:
1) The overall architecture (encoder-decoder vs decoder-only).
2) Self-attention and multi-head attention, with a small concrete example.
3) Positional encodings and why they are needed.
4) Why Transformers parallelize better than RNNs.

Write the answer in clear sections with short paragraphs rather than bullets."""

long_prompt = """SYSTEM DESIGN NOTES:
- We are building a privacy-preserving LLM inference system with a gateway, orchestrator, and GPU workers.
- The workers host a Qwen2.5-1.5B-Instruct model in FP16 with flash attention enabled.
- Requests are encrypted with hybrid RSA+AES between orchestrator and worker, and the worker streams encrypted tokens back.
- Metrics are collected at the worker: time_to_first_token_ms, total_latency_ms, input_tokens, output_tokens, and tokens_per_second.
- We are experimenting with different configuration knobs: quantization mode, max generation tokens, and prompt length.

PERFORMANCE CONTEXT:
- For previous experiments, we observed that FP16 outperformed our naive INT8 setup because the FP16 kernels are highly optimized on this GPU, while the 8-bit path introduced some overhead.
- Now we want to understand how prompt length affects time-to-first-token and overall latency while keeping generation length and model settings fixed.

ADDITIONAL CONTEXT:
- This system is inspired by secure LLM serving designs used in confidential compute / TEE environments.
- The goal is not just raw speed, but predictable latency under privacy constraints.
- We also plan to explore multi-tenant rate limiting and worker pooling in future experiments.

QUESTION:
Given all of this context, explain in detail how Transformer models work. Cover:
1) The high-level architecture (encoder-decoder vs decoder-only) and how those variants are used in practice.
2) Self-attention and multi-head attention, with a concrete toy example of how attention weights are computed.
3) Positional encodings: what problem they solve, and the difference between sinusoidal and learned embeddings.
4) Why Transformers enable much better parallelization than RNNs on modern GPUs.
5) How these architectural properties interact with system-level concerns like throughput, latency, and memory usage when deploying LLMs in production.

Write the answer as a structured explanation with clear section headings and 2–4 short paragraphs per section."""

for name, prompt in [("short", short_prompt),
                     ("medium", medium_prompt),
                     ("long", long_prompt)]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(name, "tokens:", ids.shape[1])