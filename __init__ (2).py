"""
Provider Service
Unified abstraction over Claude (Anthropic SDK) and Llama (Ollama / local).
All routers call this; swapping providers is a single config field.
"""

from __future__ import annotations
import os
import json
import httpx
from typing import AsyncIterator

from app.models.schemas import Message, ProviderConfig


# ── Default models ────────────────────────────────────────────────────────────

DEFAULTS = {
    "claude": "claude-sonnet-4-20250514",
    "llama":  "llama3.1:8b",
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model(cfg: ProviderConfig) -> str:
    return cfg.model or DEFAULTS[cfg.provider]


def _anthropic_headers() -> dict:
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content}
            for m in messages if m.role != "system"]


def _extract_system(messages: list[Message]) -> str | None:
    for m in messages:
        if m.role == "system":
            return m.content
    return None


# ── Core completion ───────────────────────────────────────────────────────────

async def complete(
    messages: list[Message],
    cfg: ProviderConfig,
    system: str | None = None,
) -> tuple[str, dict]:
    """
    Returns (reply_text, usage_dict).
    Supports Claude and Llama (via Ollama).
    """
    model = _resolve_model(cfg)
    sys_prompt = system or _extract_system(messages)

    if cfg.provider == "claude":
        return await _claude_complete(messages, model, cfg, sys_prompt)
    else:
        return await _llama_complete(messages, model, cfg, sys_prompt)


async def _claude_complete(
    messages: list[Message],
    model: str,
    cfg: ProviderConfig,
    system: str | None,
) -> tuple[str, dict]:
    payload: dict = {
        "model": model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "messages": _to_anthropic_messages(messages),
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=_anthropic_headers(),
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

    text = "".join(
        b["text"] for b in data.get("content", []) if b.get("type") == "text"
    )
    usage = data.get("usage", {})
    return text, usage


async def _llama_complete(
    messages: list[Message],
    model: str,
    cfg: ProviderConfig,
    system: str | None,
) -> tuple[str, dict]:
    ollama_msgs = []
    if system:
        ollama_msgs.append({"role": "system", "content": system})
    for m in messages:
        if m.role != "system":
            ollama_msgs.append({"role": m.role, "content": m.content})

    payload = {
        "model": model,
        "messages": ollama_msgs,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    text = data.get("message", {}).get("content", "")
    usage = {
        "input_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
    }
    return text, usage


# ── Streaming completion ──────────────────────────────────────────────────────

async def stream_complete(
    messages: list[Message],
    cfg: ProviderConfig,
    system: str | None = None,
) -> AsyncIterator[str]:
    """Yields text chunks. Use with StreamingResponse."""
    model = _resolve_model(cfg)
    sys_prompt = system or _extract_system(messages)

    if cfg.provider == "claude":
        async for chunk in _claude_stream(messages, model, cfg, sys_prompt):
            yield chunk
    else:
        async for chunk in _llama_stream(messages, model, cfg, sys_prompt):
            yield chunk


async def _claude_stream(messages, model, cfg, system):
    payload = {
        "model": model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "messages": _to_anthropic_messages(messages),
        "stream": True,
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers=_anthropic_headers(),
            json=payload,
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    try:
                        evt = json.loads(line[6:])
                        if evt.get("type") == "content_block_delta":
                            yield evt["delta"].get("text", "")
                    except json.JSONDecodeError:
                        pass


async def _llama_stream(messages, model, cfg, system):
    ollama_msgs = []
    if system:
        ollama_msgs.append({"role": "system", "content": system})
    for m in messages:
        if m.role != "system":
            ollama_msgs.append({"role": m.role, "content": m.content})

    payload = {
        "model": model,
        "messages": ollama_msgs,
        "stream": True,
        "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST", f"{OLLAMA_URL}/api/chat", json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        pass
