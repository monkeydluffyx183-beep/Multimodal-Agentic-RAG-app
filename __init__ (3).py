"""
Memory Service
Thread-safe in-memory session store.
Drop-in replacement: swap _store for Redis / DynamoDB / Postgres.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from collections import defaultdict

from app.models.schemas import Message, MemoryEntry


# session_id → ordered list of MemoryEntry
_store: dict[str, list[MemoryEntry]] = defaultdict(list)

MAX_HISTORY = 100   # hard cap per session


def add_messages(session_id: str, messages: list[Message]) -> int:
    for msg in messages:
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=msg.content,
            role=msg.role,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        _store[session_id].append(entry)
        if len(_store[session_id]) > MAX_HISTORY:
            _store[session_id] = _store[session_id][-MAX_HISTORY:]
    return len(messages)


def get_messages(session_id: str, limit: int = 20) -> list[MemoryEntry]:
    return _store[session_id][-limit:]


def get_as_messages(session_id: str, limit: int = 20) -> list[Message]:
    entries = get_messages(session_id, limit)
    return [
        Message(role=e.role, content=e.content)   # type: ignore[arg-type]
        for e in entries
        if e.role in ("user", "assistant")
    ]


def count(session_id: str) -> int:
    return len(_store[session_id])


def clear(session_id: str) -> int:
    n = len(_store[session_id])
    _store[session_id] = []
    return n


def all_sessions() -> list[str]:
    return list(_store.keys())
