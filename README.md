"""
Memory Router
POST /memory/add        — persist messages to a session
POST /memory/chat       — stateful chat with session history
GET  /memory/sessions/{session_id} — retrieve history
DELETE /memory/sessions/{session_id} — clear a session
GET  /memory/sessions   — list all sessions
"""

from __future__ import annotations
import uuid
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    MemoryAddRequest, MemoryGetResponse,
    MemoryChatRequest, MemoryChatResponse,
    Message,
)
from app.services import memory_service
from app.services.provider import complete

router = APIRouter()


@router.post("/add")
async def add_messages(req: MemoryAddRequest):
    n = memory_service.add_messages(req.session_id, req.messages)
    return {
        "session_id": req.session_id,
        "added": n,
        "total": memory_service.count(req.session_id),
    }


@router.post("/chat", response_model=MemoryChatResponse)
async def memory_chat(req: MemoryChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    model = req.config.model or (
        "claude-sonnet-4-20250514" if req.config.provider == "claude" else "llama3.1:8b"
    )

    # Build full history
    history = memory_service.get_as_messages(session_id)
    history.append(Message(role="user", content=req.message))

    system = req.system_prompt or (
        "You are a helpful assistant with memory of the ongoing conversation."
    )
    reply, usage = await complete(history, req.config, system)

    # Persist both turns
    memory_service.add_messages(session_id, [
        Message(role="user", content=req.message),
        Message(role="assistant", content=reply),
    ])

    return MemoryChatResponse(
        provider=req.config.provider,
        model=model,
        session_id=session_id,
        reply=reply,
        history_length=memory_service.count(session_id),
        usage=usage,
    )


@router.get("/sessions/{session_id}", response_model=MemoryGetResponse)
async def get_session(session_id: str, limit: int = 20):
    msgs = memory_service.get_messages(session_id, limit)
    if not msgs:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or empty")
    return MemoryGetResponse(
        session_id=session_id,
        messages=msgs,
        total=memory_service.count(session_id),
    )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    n = memory_service.clear(session_id)
    return {"session_id": session_id, "cleared": n}


@router.get("/sessions")
async def list_sessions():
    sessions = memory_service.all_sessions()
    return {
        "sessions": [
            {"session_id": s, "message_count": memory_service.count(s)}
            for s in sessions
        ]
    }
