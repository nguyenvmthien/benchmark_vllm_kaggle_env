"""OpenAI-compatible streaming HTTP client."""
from __future__ import annotations
import asyncio
import json
import time
from dataclasses import asdict, dataclass
import aiohttp

@dataclass(slots=True)
class RequestResult:
    prompt: str
    success: bool
    ttft: float | None
    latency: float
    output_tokens: int
    itl: float | None
    error: str | None = None
    token_count_source: str = "usage"

    def to_dict(self) -> dict:
        return asdict(self)

async def fetch_request(session: aiohttp.ClientSession, api_url: str, model: str,
                        prompt: str, max_tokens: int, *, timeout: float = 120.0) -> dict:
    """Measure one stream. ITL is mean time per output token (also called TPOT)."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    started = time.perf_counter()
    first_token_at: float | None = None
    content_events = 0
    usage_tokens: int | None = None
    try:
        async with session.post(api_url, json=payload, timeout=timeout) as response:
            if response.status != 200:
                body = (await response.text())[:500]
                return RequestResult(prompt, False, None, time.perf_counter() - started, 0,
                                     None, f"HTTP {response.status}: {body}").to_dict()
            buffer = b""
            async for chunk in response.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        usage_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or []
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content:
                        content_events += 1
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return RequestResult(prompt, False, None, time.perf_counter() - started, 0, None,
                             f"{type(exc).__name__}: {exc}").to_dict()
    except Exception as exc:
        return RequestResult(prompt, False, None, time.perf_counter() - started, 0, None,
                             f"{type(exc).__name__}: {exc}").to_dict()
    ended = time.perf_counter()
    output_tokens = usage_tokens if usage_tokens is not None else content_events
    source = "usage" if usage_tokens is not None else "content_events_estimate"
    ttft = first_token_at - started if first_token_at is not None else None
    itl = ((ended - first_token_at) / (output_tokens - 1)
           if first_token_at is not None and output_tokens > 1 else None)
    success = ttft is not None and output_tokens > 0
    return RequestResult(prompt, success, ttft, ended - started, output_tokens, itl,
                         None if success else "Response contained no output tokens", source).to_dict()
