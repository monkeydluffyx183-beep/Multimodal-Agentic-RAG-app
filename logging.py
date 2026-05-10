"""
LLM Hub — Unified API Backend
Supports: Claude (Anthropic) + Llama (Local/Ollama)
Features: Agents, RAG, Multi-Agent, Voice, MCP, Memory
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.middleware.logging import LoggingMiddleware
from app.routers import agents, rag, multi_agent, voice, mcp, memory, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 LLM Hub starting up...")
    yield
    print("🛑 LLM Hub shutting down...")


app = FastAPI(
    title="LLM Hub API",
    description="Unified backend for Claude + Llama — Agents, RAG, Multi-Agent, Voice, MCP, Memory",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# Routers
app.include_router(health.router,       tags=["Health"])
app.include_router(agents.router,       prefix="/agents",      tags=["Agents"])
app.include_router(rag.router,          prefix="/rag",         tags=["RAG"])
app.include_router(multi_agent.router,  prefix="/multi-agent", tags=["Multi-Agent"])
app.include_router(voice.router,        prefix="/voice",       tags=["Voice"])
app.include_router(mcp.router,          prefix="/mcp",         tags=["MCP"])
app.include_router(memory.router,       prefix="/memory",      tags=["Memory"])
