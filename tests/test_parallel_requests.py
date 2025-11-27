# tests/test_parallel_requests.py

import asyncio
import aiohttp
import os

GATEWAY_URL = "http://localhost:8000/infer"

REQUEST_PAYLOAD = {
    "tenant_id": "tenant1",
    "prompt": "hello world",
    "mode": "plain",
    "region": "us-west-1",
    "client_pubkey": ""
}

# Number of concurrent requests to test
PARALLEL_REQUESTS = 5


async def send_request(i):
    async with aiohttp.ClientSession() as session:
        async with session.post(GATEWAY_URL, json=REQUEST_PAYLOAD) as resp:
            print(f"[Request {i}] Status:", resp.status)

            # Stream framed binary tokens
            async for chunk in resp.content:
                print(f"[Request {i}] Received {len(chunk)} bytes")

    print(f"[Request {i}] Completed")


async def main():
    tasks = [send_request(i) for i in range(PARALLEL_REQUESTS)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
