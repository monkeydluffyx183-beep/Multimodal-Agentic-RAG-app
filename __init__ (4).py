"""
Agents Router
POST /agents/run        — single-turn agent with optional tool loop
POST /agents/stream     — streaming agent response
GET  /agents/tools      — list available tools
"""

from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AgentRunRequest, AgentRunResponse, ToolCall, Message,
)
from app.services.provider import complete, stream_complete

router = APIRouter()

# ── Built-in tool registry ────────────────────────────────────────────────────

TOOLS: dict[str, dict] = {
    "web_search": {
        "description": "Search the web for current information.",
        "input_schema": {"query": "string"},
    },
    "calculator": {
        "description": "Evaluate a mathematical expression.",
        "input_schema": {"expression": "string"},
    },
    "summarizer": {
        "description": "Summarise a long piece of text.",
        "input_schema": {"text": "string"},
    },
    "code_executor": {
        "description": "Run Python code and return stdout.",
        "input_schema": {"code": "string"},
    },
}


def _mock_tool_call(name: str, inp: dict) -> str:
    """Simulated tool execution (replace with real implementations)."""
    if name == "calculator":
        try:
            return str(eval(inp.get("expression", "0"), {"__builtins__": {}}))
        except Exception as e:
            return f"Error: {e}"
    if name == "web_search":
        return f"[Simulated search results for: {inp.get('query')}]"
    if name == "summarizer":
        text = inp.get("text", "")
        return f"Summary ({len(text)} chars): {text[:120]}..."
    if name == "code_executor":
        return "[Code execution sandbox not active in this deployment]"
    return f"[Tool '{name}' executed with input: {inp}]"


def _build_system(tools: list[str], base: str | None) -> str:
    parts = [base or "You are a helpful AI agent."]
    if tools:
        tool_list = "\n".join(
            f"- {t}: {TOOLS[t]['description']}"
            for t in tools if t in TOOLS
        )
        parts.append(
            f"\nAvailable tools:\n{tool_list}\n\n"
            "When you need a tool, reply with a JSON block:\n"
            '```tool\n{"name": "<tool>", "input": {<args>}}\n```\n'
            "After you receive the tool result, continue your reasoning."
        )
    return "\n".join(parts)


@router.get("/tools")
async def list_tools():
    return {"tools": [{"name": k, **v} for k, v in TOOLS.items()]}


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(req: AgentRunRequest):
    system = _build_system(req.tools, req.system_prompt)
    messages = list(req.messages)
    tool_calls: list[ToolCall] = []
    steps = 0
    model = req.config.model or (
        "claude-sonnet-4-20250514" if req.config.provider == "claude" else "llama3.1:8b"
    )

    for _ in range(5):   # max 5 tool-use iterations
        steps += 1
        reply, usage = await complete(messages, req.config, system)

        # Detect tool call in reply
        if "```tool" in reply and req.tools:
            import json, re
            match = re.search(r"```tool\s*\n(.*?)\n```", reply, re.DOTALL)
            if match:
                try:
                    call = json.loads(match.group(1))
                    name, inp = call["name"], call.get("input", {})
                    output = _mock_tool_call(name, inp)
                    tool_calls.append(ToolCall(name=name, input=inp, output=output))
                    # Feed result back
                    messages.append(Message(role="assistant", content=reply))
                    messages.append(Message(
                        role="user",
                        content=f"Tool result for {name}:\n{output}\nContinue."
                    ))
                    continue
                except Exception:
                    pass
        break   # no tool call — done

    return AgentRunResponse(
        provider=req.config.provider,
        model=model,
        reply=reply,
        tool_calls=tool_calls,
        steps=steps,
        usage=usage,
    )


@router.post("/stream")
async def stream_agent(req: AgentRunRequest):
    system = _build_system(req.tools, req.system_prompt)

    async def _gen():
        async for chunk in stream_complete(req.messages, req.config, system):
            yield chunk

    return StreamingResponse(_gen(), media_type="text/plain")
