# SecureChat

SecureChat is a small **end-to-end LLM inference service** that focuses on **privacy** and **performance**.

It simulates running a multi-tenant LLM backend on **confidential computing infrastructure**, using:

* A **Gateway** (HTTP)
* An **Orchestrator** (gRPC)
* “Enclave” **Workers** (separate processes with their own keys + GPU model)

Right now this runs locally and uses software-simulated enclaves, but it follows the same design constraints you’d use with real TEEs (Nitro Enclaves, TDX, etc.): **minimal logging**, **encrypted boundaries**, and **ephemeral workers**.

---

## Project Document

More detailed design notes and motivation are in the project doc:

> [Project Document](Project%20Document)

---

## Architecture (made with Mermaid)
<img width="863" height="161" alt="image" src="https://github.com/user-attachments/assets/3a92ae70-f344-472b-b1fa-e6948d92ffd6" />


## Getting Started
Please refer to the [Project Document](https://docs.google.com/document/d/1SiTb16rfHMk3bPADsA9GcitVUdFqDS-PdL3RZqB8ZOE/edit?usp=sharing).


High-level data flow:

**Client → HTTP (Gateway) → gRPC (Orchestrator) → gRPC (Worker “enclave”) → back as streaming encrypted tokens → Client**

### Gateway (FastAPI)

* Exposes `POST /infer`
* Accepts: `tenant_id`, `prompt`, `mode`, `region`
* Calls into the orchestrator over gRPC and streams the worker’s encrypted token frames back to the client.

### Orchestrator (gRPC server)

* Receives requests from the gateway
* Uses `InferenceDispatcher` to:
    * Normalize / assign **request\_id**
    * Encrypt the prompt for the worker using hybrid RSA+AES
* Maintains a client for one or more workers
* Runs an async loop that:
    * Calls **RunInference** on the worker
    * Frames the encrypted tokens (length-prefixed)
    * Streams them back to the gateway

### Workers (Simulated enclaves, gRPC server)

* Each worker process:
    * Generates/loads its own **RSA keypair**
    * Loads a **Qwen2.5-1.5B-Instruct** model on GPU
* Maintains an internal state machine: `MODEL_LOADING` → **READY** → `BUSY` → `STREAMING` → **READY** (or `FAILED`)
* **RunInference:**
    * Decrypts the hybrid envelope from the orchestrator:
        RSA private key → AES key → AES-GCM → plaintext prompt
    * Runs HF **generate** with **TextIteratorStreamer** for streaming
    * Encrypts each token chunk with AES-GCM and streams it back
    * Logs *only* metadata + metrics (**no plaintext**)

Architecture diagram in the repo:

> Gateway (FastAPI) → Orchestrator (gRPC) → Worker (gRPC, GPU)
> (See the Mermaid image in the repo for the visual version.)

---

## Security & Privacy Model (Current State)

This project is **not production-secure**, but it intentionally follows a TEE-style programming model.

### What it does:

* Treats worker processes as **“enclaves”**
    * Only workers see decrypted prompts and model outputs
    * Host components (gateway/orchestrator) see opaque ciphertexts
* Uses **hybrid RSA+AES** for orchestrator → worker:
    * AES-GCM for data, RSA for wrapping AES keys
* Uses **symmetric AES-GCM** for worker → orchestrator:
    * Worker encrypts each token before sending
* Enforces **privacy-aware logging**:
    * No content or decrypted data is logged
    * Only request/worker IDs, timings, token counts, and bytes

### What it does not do yet:

* No real auth (API keys/JWTs) for tenants
* No real hardware enclave / attestation

The intention is to show the **shapes** and tradeoffs (encryption boundaries, metrics, ephemeral workers), not to ship a production-ready secure system.

## Performance Experiments & Results

This section summarizes the performance experiments run on the **SecureLLM prototype** and the main observations, grounded strictly in measured data.

---

### Measurement Setup

* **Model:** Qwen/Qwen2.5-1.5B-Instruct
* **Hardware:** Single **NVIDIA GPU** (CUDA available)
* **Worker:**
    * Hugging Face `AutoModelForCausalLM`
    * Quantization: `MODEL_QUANTIZATION_MODE` $\in$ {**INT8**, **FP16**}
    * Attention backend: `attn_implementation` $\in$ {"**flash\_attention\_2**", "**sdpa**"}
    * Streaming via `TextIteratorStreamer`
* **System topology:**
    * HTTPS **Gateway** $\to$ gRPC **Orchestrator** $\to$ gRPC **Workers** 
    * Orchestrator $\leftrightarrow$ Worker payloads encrypted with **hybrid RSA + AES-GCM**
* **Worker metrics (per request):**
    * `ttft_ms` – **time-to-first-token** at the worker
    * `total_ms` – worker time from request start to last token
    * `input_tokens`, `output_tokens`
    * `tokens_per_sec` = `output_tokens` / (`total_ms` / 1000)
* **Gateway metrics (per request):**
    * `total_ms` – end-to-end HTTP latency
    * `frames` – token frames sent to client
    * `bytes` – total bytes streamed

> All logs are **content-free**; only metadata and timings are recorded.

---

### 1. Quantization: INT8 vs FP16

We compared **INT8 (bitsandbytes)** vs **FP16** on the same medium-length prompt ($\approx$154 tokens), varying `MAX_GENERATION_TOKENS`.

| Dtype | Max gen tokens | input\_tokens | output\_tokens | ttft\_ms | total\_ms | tokens/sec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| INT8 | 128 | 154 | 104 | 433 | 7,082 | 14.69 |
| INT8 | 256 | 154 | 202 | 466 | 14,446 | 13.98 |
| INT8 | 512 | 154 | 342 | 492 | 29,881 | 11.45 |
| FP16 | 128 | 154 | 105 | 442 | 2,408 | **43.60** |
| FP16 | 256 | 154 | 211 | 438 | 4,854 | **43.47** |
| FP16 | 512 | 154 | 419 | 400 | 11,274 | 37.17 |

**Observation**

In this setup, **FP16 is substantially faster than INT8**:

* At 256 max tokens: $\approx$43.5 tok/s (FP16) vs $\approx$14 tok/s (INT8).
* TTFT is similar in both modes (hundreds of ms).

**Conclusion for this stack:** bitsandbytes INT8 behaves primarily as a **memory optimization**, not a latency optimization, while FP16 leverages highly optimized GPU kernels.

---

### 2. Prompt Length: Short vs Medium vs Long

We tested three prompts: Short (21 tokens), Medium (159 tokens), and Long (408 tokens).

*Configuration: FP16, `MAX_GENERATION_TOKENS = 512`, FlashAttention enabled, single worker.*

| Prompt type | input\_tokens | output\_tokens | ttft\_ms | total\_ms | tokens/sec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Short** | 21 | 196 | 67 | 3,108 | **63.06** |
| **Medium** | 159 | 302 | 70 | 10,978 | 27.51 |
| **Long** | 408 | 335 | 80 | 14,998 | 22.34 |

**Observation**

* TTFT is almost **flat** across prompt sizes (67–80 ms).
* Total latency and tokens/sec degrade as prompt + output length grow: Throughput drops from $\approx$63 tok/s (short) $\to$ $\approx$27.5 (medium) $\to$ $\approx$22.3 (long).

**Conclusion:** This matches the expectation that longer contexts increase **attention cost** per decoding step.

---

### 3. Attention Backend: FlashAttention2 vs SDPA

For the long prompt (408 tokens), FP16, `MAX_GENERATION_TOKENS = 512`:

| Attention backend | input\_tokens | output\_tokens | ttft\_ms | total\_ms | tokens/sec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FlashAttention2** | 408 | 335 | 403 | 15,327 | 21.86 |
| **SDPA** | 408 | 365 | 416 | 14,492 | **25.19** |

**Observation**

SDPA produced more tokens (365 vs 335) and still had lower total latency and higher tokens/sec. For this model and GPU, **FlashAttention2 did not show a clear performance advantage**; SDPA was slightly better in this particular measurement.

---

### 4. Worker vs End-to-End Latency

Comparing worker and gateway timings on the long prompt:

| Config | Worker total\_ms | Gateway total\_ms | Overhead |
| :--- | :--- | :--- | :--- |
| FlashAttention ON | 15,327 | 15,372 | ~45 ms |
| FlashAttention OFF (SDPA) | 14,492 | 14,531 | ~39 ms |

**Observation**

Gateway + orchestrator overhead is **< 1%** of total latency (~40–45 ms added to ~15 s). The system is clearly **model-bound**, not network/serialization-bound.

---

### 5. Concurrency on a Single Worker

We used a heavier prompt (~248 input tokens, 395 output tokens) and varied client-side concurrency (all requests hit a single worker).

#### Worker-side metrics

| Parallel clients | worker total\_ms (per req) | tokens/sec | input\_tokens | output\_tokens |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 11,420 | 34.59 | 248 | 395 |
| 2 | ~11,366 (11,355 & 11,378) | ~34.7 | 248 | 395 |
| 4 | ~11,394 (across 4 reqs) | ~$34.7 | 248 | 395 |

#### Gateway end-to-end metrics

| Parallel clients | gateway total\_ms per request |
| :--- | :--- |
| 1 | ~11,457 |
| 2 | ~22,807 |
| 4 | ~45,722 |

**Observation**

* Per-request worker latency is almost **constant** as concurrency increases (~11.4s per request).
* End-to-end latency scales approximately **linearly** with concurrency for a single worker.

**Conclusion:** With one worker, system throughput is **fixed** by the single GPU; extra clients don’t increase throughput, only **queueing**.

---

### 6. Multi-Worker Scaling (2 Workers)

We ran two workers on the same GPU with simple **round-robin** dispatching from the orchestrator.

*Prompt here: $\approx$6 input\_tokens, 374 output\_tokens.*

#### Worker metrics (4 concurrent requests total)

| Worker | total\_ms (per req) | tokens/sec | input\_tokens | output\_tokens |
| :--- | :--- | :--- | :--- | :--- |
| worker-1 | 7,899 / 7,424 | ~47–50 | 6 | 374 |
| worker-2 | 7,816 / 7,416 | ~47–50 | 6 | 374 |

#### Gateway metrics

4 concurrent requests completed in: **$\approx$30,712 ms** total.

**Observation**

In the single-worker test with 4 concurrent requests (different prompt, ~45.7s), the time to completion was higher. Here, adding a second worker process **reduced the time to finish 4 requests** by roughly ~1.5$\times$** (~45.7s $\to$ ~30.7s) in this small-scale test, by improving concurrency and scheduling.

---

### 7. Summary of Findings

* **FP16 vs INT8:** On this stack (HF + bitsandbytes), **FP16 clearly outperforms INT8** in tokens/sec ($\approx$3$\times$ at 256 max tokens), with similar TTFT. INT8 behaves as a memory optimization, not a latency optimization, in this configuration.
* **Sequence length matters:** Longer prompts and generations increase **total latency** and reduce tokens/sec, while TTFT remains relatively stable.
* **FlashAttention2 vs SDPA:** SDPA slightly outperformed FlashAttention2 in the measured run (lower latency, higher tokens/sec). **No clear performance win for FA2** was observed at this scale.
* **System overhead is negligible:** Gateway + orchestrator add $\approx$40–45 ms on top of $\approx$14–15 s worker time. The overall system is **compute-bound** on the LLM.
* **Concurrency vs capacity:** With a single worker, increasing concurrent clients increases per-request end-to-end latency almost linearly. **Adding a second worker** significantly improves the time to complete multiple requests, even on the same GPU.


## What’s Implemented vs Planned

### Implemented in this repo:

* **End-to-end pipeline:**
    **FastAPI Gateway** → gRPC **Orchestrator** → gRPC **Worker(s)** → streaming response back to client.
* **“Enclave” workers:**
    Each worker is its own process with:
    * **Qwen/Qwen2.5-1.5B-Instruct** on GPU
    * Configurable quantization: FP16, bitsandbytes INT8
    * Configurable attention: FlashAttention (if installed) or PyTorch SDPA
* **Hybrid crypto on the host ↔ worker boundary:**
    * Orchestrator → Worker: hybrid **RSA + AES-GCM** envelope for the prompt
    * Worker → Orchestrator/Gateway: **AES-GCM** token streaming back
* **Privacy-aware logging:**
    * No prompts, outputs, or decrypted content ever logged
    * Only metadata: **request\_id**, **worker\_id**, timings, token counts, bytes
* **Basic multi-worker support:**
    You can run multiple worker processes (different ports) and dispatch requests across them
* **Metrics:**
    * Worker: **time-to-first-token (TTFT)**, total latency, input/output tokens, tokens/sec
    * Gateway: **end-to-end request latency**, frames, bytes


---


## Running Locally 
### Create environment and install deps
```bash
conda create -n securellm python=3.11
conda activate securellm

pip install -r requirements.txt
```

### Check GPU

You need a CUDA-capable GPU for the worker process.

### Start workers

Single worker:

```bash
python src/workers/worker_server.py --worker_id worker-1 --port 50051
```
### Start orchestrator
```bash
python src/orchestrator/orchestrator_server.py
```

### Start Gateway
```bash
python src/gateway/gateway_server.py
```

### Send a test request
```bash
curl -N -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant1",
    "prompt": "You are an expert ML systems engineer. Explain how transformer-based LLMs work and how to optimize their inference performance.",
    "mode": "plain",
    "region": "us-west-1"
  }' --output out.bin
```

You should see logs from the gateway, orchestrator, and worker. out.bin will contain the streamed encrypted bytes (in this prototype we don’t decrypt them back on the client).

### InProgress :

* Real RAG pipeline (embedding + vector DB + mode=“rag” path)
* Real attestation + measurement + key provisioning
* Real tenant authentication (API keys / JWTs) and per-tenant policy
* Real hardware TEEs (this is all single-machine, dev setup)



