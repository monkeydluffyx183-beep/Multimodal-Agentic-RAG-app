"""Health check router."""

import os
import httpx
from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()

@router.get("/health")
async def health():
    checks: dict = {}

    # Claude
    checks["claude"] = bool(os.getenv("ANTHROPIC_API_KEY"))

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/tags")
            checks["ollama"] = r.status_code == 200
    except Exception:
        checks["ollama"] = False

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": checks,
        "version": "1.0.0",
    }

@router.get("/")
async def root():
    return {
        "name": "LLM Hub API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/agents", "/rag", "/multi-agent", "/voice", "/mcp", "/memory"],
    }
