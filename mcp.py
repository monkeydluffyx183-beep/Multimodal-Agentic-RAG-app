"""
RAG Service
Lightweight in-memory vector store using numpy cosine similarity.
Swap out _embed() for OpenAI / local embeddings as needed.
Supports: basic, agentic, corrective, hybrid retrieval modes.
"""

from __future__ import annotations
import os
import re
import math
import uuid
import hashlib
from typing import Any
import httpx

from app.models.schemas import (
    ProviderConfig, Message,
    RAGSource,
)
from app.services.provider import complete


# ── Embedding ─────────────────────────────────────────────────────────────────
# Uses OpenAI's text-embedding-3-small when OPENAI_API_KEY is set,
# otherwise falls back to a fast TF-IDF-style sparse vector.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_DIM = 256  # used only for fallback


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _sparse_embed(text: str) -> list[float]:
    """Deterministic sparse bag-of-words vector (no external deps)."""
    tokens = _tokenize(text)
    vec = [0.0] * EMBED_DIM
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _embed(text: str) -> list[float]:
    if OPENAI_API_KEY:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": "text-embedding-3-small", "input": text},
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
    return _sparse_embed(text)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


# ── In-memory store ───────────────────────────────────────────────────────────

_store: dict[str, list[dict]] = {}   # collection → [{ id, text, embedding, metadata }]


async def ingest(
    texts: list[str],
    metadata: list[dict[str, Any]] | None,
    collection: str,
) -> int:
    if collection not in _store:
        _store[collection] = []
    metas = metadata or [{} for _ in texts]
    for text, meta in zip(texts, metas):
        emb = await _embed(text)
        _store[collection].append({
            "id": str(uuid.uuid4()),
            "text": text,
            "embedding": emb,
            "metadata": meta,
        })
    return len(texts)


async def retrieve(
    query: str,
    collection: str,
    top_k: int,
) -> list[RAGSource]:
    docs = _store.get(collection, [])
    if not docs:
        return []
    q_emb = await _embed(query)
    scored = [
        (doc, _cosine(q_emb, doc["embedding"]))
        for doc in docs
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        RAGSource(text=d["text"], score=round(s, 4), metadata=d["metadata"])
        for d, s in scored[:top_k]
    ]


# ── RAG modes ─────────────────────────────────────────────────────────────────

async def rag_query(
    query: str,
    collection: str,
    top_k: int,
    mode: str,
    cfg: ProviderConfig,
) -> tuple[str, list[RAGSource]]:
    if mode == "basic":
        return await _basic_rag(query, collection, top_k, cfg)
    elif mode == "agentic":
        return await _agentic_rag(query, collection, top_k, cfg)
    elif mode == "corrective":
        return await _corrective_rag(query, collection, top_k, cfg)
    elif mode == "hybrid":
        return await _hybrid_rag(query, collection, top_k, cfg)
    return await _basic_rag(query, collection, top_k, cfg)


async def _basic_rag(query, collection, top_k, cfg):
    sources = await retrieve(query, collection, top_k)
    context = "\n\n".join(f"[{i+1}] {s.text}" for i, s in enumerate(sources))
    system = (
        "You are a helpful assistant. Answer the user's question using only "
        "the provided context. If the context is insufficient, say so.\n\n"
        f"CONTEXT:\n{context}"
    )
    reply, _ = await complete([Message(role="user", content=query)], cfg, system)
    return reply, sources


async def _agentic_rag(query, collection, top_k, cfg):
    # Step 1: LLM refines the query
    refine_prompt = (
        f"Rewrite this query to maximise retrieval relevance. "
        f"Return ONLY the rewritten query, nothing else.\nQuery: {query}"
    )
    refined, _ = await complete([Message(role="user", content=refine_prompt)], cfg)
    refined = refined.strip()

    sources = await retrieve(refined, collection, top_k)
    context = "\n\n".join(f"[{i+1}] {s.text}" for i, s in enumerate(sources))
    system = (
        "You are an agentic research assistant. Synthesise an answer from the context.\n\n"
        f"CONTEXT:\n{context}"
    )
    reply, _ = await complete([Message(role="user", content=query)], cfg, system)
    return reply, sources


async def _corrective_rag(query, collection, top_k, cfg):
    sources = await retrieve(query, collection, top_k)

    # Grade each source
    good_sources = []
    for src in sources:
        grade_prompt = (
            f"Does this passage help answer the question?\n"
            f"Question: {query}\nPassage: {src.text}\n"
            f"Reply YES or NO only."
        )
        verdict, _ = await complete([Message(role="user", content=grade_prompt)], cfg)
        if "YES" in verdict.upper():
            good_sources.append(src)

    if not good_sources:
        good_sources = sources[:2]   # fallback: use top-2

    context = "\n\n".join(f"[{i+1}] {s.text}" for i, s in enumerate(good_sources))
    system = (
        "You are a precise assistant. Use only high-quality sources to answer.\n\n"
        f"VERIFIED CONTEXT:\n{context}"
    )
    reply, _ = await complete([Message(role="user", content=query)], cfg, system)
    return reply, good_sources


async def _hybrid_rag(query, collection, top_k, cfg):
    # Dense retrieval
    dense = await retrieve(query, collection, top_k)

    # Sparse BM25-style: count keyword overlap
    q_tokens = set(_tokenize(query))
    docs = _store.get(collection, [])

    def bm25_score(doc_text: str) -> float:
        d_tokens = _tokenize(doc_text)
        overlap = sum(1 for t in d_tokens if t in q_tokens)
        return overlap / (len(d_tokens) + 1e-9)

    sparse_scored = sorted(docs, key=lambda d: bm25_score(d["text"]), reverse=True)
    sparse = [
        RAGSource(text=d["text"], score=round(bm25_score(d["text"]), 4), metadata=d["metadata"])
        for d in sparse_scored[:top_k]
    ]

    # Merge & deduplicate by text
    seen, merged = set(), []
    for s in [*dense, *sparse]:
        if s.text not in seen:
            seen.add(s.text)
            merged.append(s)

    context = "\n\n".join(f"[{i+1}] {s.text}" for i, s in enumerate(merged[:top_k]))
    system = (
        "You are a research assistant using hybrid-search retrieved documents.\n\n"
        f"CONTEXT:\n{context}"
    )
    reply, _ = await complete([Message(role="user", content=query)], cfg, system)
    return reply, merged[:top_k]
