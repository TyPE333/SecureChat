# tests/test_parallel_requests.py

import asyncio
import aiohttp
import os
import time  # ← add this

GATEWAY_URL = "http://localhost:8000/infer"

REQUEST_PAYLOAD = {
    "tenant_id": "tenant1",
    "prompt": """ ... your long prompt ... """,
    "mode": "plain",
    "region": "us-west-1",
    "client_pubkey": ""
}

NUM_REQUESTS = int(os.getenv("PARALLEL_REQUESTS", "4"))


async def send_request(i: int):
    async with aiohttp.ClientSession() as session:
        async with session.post(GATEWAY_URL, json=REQUEST_PAYLOAD) as resp:
            print(f"[Request {i}] Status:", resp.status)
            async for chunk in resp.content:
                # per-chunk logging is noisy but fine for now
                pass
    print(f"[Request {i}] Completed")


async def main():
    start = time.perf_counter()
    tasks = [send_request(i) for i in range(NUM_REQUESTS)]
    await asyncio.gather(*tasks)
    end = time.perf_counter()
    print(f"\n== Completed {NUM_REQUESTS} requests in {(end - start)*1000:.1f} ms ==")


if __name__ == "__main__":
    asyncio.run(main())