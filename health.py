"""
MCP Router — Model Context Protocol integration stubs
POST /mcp/tools        — list tools from registered servers
POST /mcp/call         — invoke a specific MCP tool
POST /mcp/agent        — run an agent with MCP tool access
GET  /mcp/servers      — list registered MCP servers
"""

from __future__ import annotations
import time
import json
import re
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    MCPTool, MCPCallRequest, MCPCallResponse,
    MCPAgentRequest, MCPAgentResponse,
    ToolCall, Message,
)
from app.services.provider import complete

router = APIRouter()

# ── MCP server registry ────────────────────────────────────────────────────────
# Each entry maps to an MCP server. Replace stubs with real MCP client calls
# (e.g. via mcp-python SDK: https://github.com/modelcontextprotocol/python-sdk)

MCP_SERVERS: dict[str, dict] = {
    "github": {
        "description": "GitHub — issues, PRs, code search",
        "tools": [
            MCPTool(name="list_issues", description="List open issues for a repo",
                    input_schema={"owner": "string", "repo": "string"}),
            MCPTool(name="create_issue", description="Open a new issue",
                    input_schema={"owner": "string", "repo": "string", "title": "string", "body": "string"}),
            MCPTool(name="search_code", description="Search code in a repo",
                    input_schema={"query": "string", "repo": "string"}),
        ],
    },
    "notion": {
        "description": "Notion — pages, databases",
        "tools": [
            MCPTool(name="search_pages", description="Search Notion pages",
                    input_schema={"query": "string"}),
            MCPTool(name="create_page", description="Create a new Notion page",
                    input_schema={"title": "string", "content": "string", "parent_id": "string"}),
        ],
    },
    "browser": {
        "description": "Browser automation via Playwright",
        "tools": [
            MCPTool(name="navigate", description="Navigate to a URL",
                    input_schema={"url": "string"}),
            MCPTool(name="click", description="Click an element",
                    input_schema={"selector": "string"}),
            MCPTool(name="extract_text", description="Extract visible page text",
                    input_schema={}),
        ],
    },
    "filesystem": {
        "description": "Read/write local files",
        "tools": [
            MCPTool(name="read_file", description="Read a file's contents",
                    input_schema={"path": "string"}),
            MCPTool(name="write_file", description="Write content to a file",
                    input_schema={"path": "string", "content": "string"}),
            MCPTool(name="list_dir", description="List files in a directory",
                    input_schema={"path": "string"}),
        ],
    },
}


def _mock_mcp_call(server: str, tool_name: str, tool_input: dict) -> str:
    """
    Stub executor. Replace with real MCP client:
        from mcp import ClientSession
        async with ClientSession(...) as session:
            result = await session.call_tool(tool_name, tool_input)
    """
    return (
        f"[MCP stub] Server='{server}' Tool='{tool_name}' "
        f"Input={json.dumps(tool_input)} → result placeholder"
    )


@router.get("/servers")
async def list_servers():
    return {
        "servers": [
            {"name": k, "description": v["description"], "tool_count": len(v["tools"])}
            for k, v in MCP_SERVERS.items()
        ]
    }


@router.get("/tools")
async def list_tools(server: str | None = None):
    if server:
        if server not in MCP_SERVERS:
            raise HTTPException(status_code=404, detail=f"Server '{server}' not found")
        return {"server": server, "tools": MCP_SERVERS[server]["tools"]}
    all_tools = []
    for srv, data in MCP_SERVERS.items():
        for tool in data["tools"]:
            all_tools.append({"server": srv, **tool.model_dump()})
    return {"tools": all_tools}


@router.post("/call", response_model=MCPCallResponse)
async def call_tool(req: MCPCallRequest):
    if req.server not in MCP_SERVERS:
        raise HTTPException(status_code=404, detail=f"Server '{req.server}' not registered")
    server_tools = [t.name for t in MCP_SERVERS[req.server]["tools"]]
    if req.tool_name not in server_tools:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found on server '{req.server}'")

    t0 = time.perf_counter()
    result = _mock_mcp_call(req.server, req.tool_name, req.tool_input)
    duration = round((time.perf_counter() - t0) * 1000, 2)

    return MCPCallResponse(
        tool_name=req.tool_name,
        server=req.server,
        result=result,
        duration_ms=duration,
    )


@router.post("/agent", response_model=MCPAgentResponse)
async def mcp_agent(req: MCPAgentRequest):
    model = req.config.model or (
        "claude-sonnet-4-20250514" if req.config.provider == "claude" else "llama3.1:8b"
    )

    # Build tool list for system prompt
    available: list[MCPTool] = []
    for srv in req.servers:
        if srv in MCP_SERVERS:
            available.extend(MCP_SERVERS[srv]["tools"])

    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in available
    )
    system = (
        "You are an MCP-enabled AI agent. Use the tools below to complete the task.\n"
        "When you need a tool, reply with:\n"
        '```tool\n{"name": "<tool_name>", "input": {<args>}}\n```\n\n'
        f"Available tools:\n{tool_descriptions}"
    )

    messages = [Message(role="user", content=req.task)]
    tool_calls_log: list[ToolCall] = []

    for _ in range(6):
        reply, usage = await complete(messages, req.config, system)

        if "```tool" in reply:
            match = re.search(r"```tool\s*\n(.*?)\n```", reply, re.DOTALL)
            if match:
                try:
                    call = json.loads(match.group(1))
                    tname, tinput = call["name"], call.get("input", {})
                    # Find which server has this tool
                    srv = next(
                        (s for s, d in MCP_SERVERS.items()
                         if any(t.name == tname for t in d["tools"])),
                        req.servers[0] if req.servers else "default"
                    )
                    output = _mock_mcp_call(srv, tname, tinput)
                    tool_calls_log.append(ToolCall(name=tname, input=tinput, output=output))
                    messages.append(Message(role="assistant", content=reply))
                    messages.append(Message(role="user", content=f"Tool result:\n{output}"))
                    continue
                except Exception:
                    pass
        break

    return MCPAgentResponse(
        provider=req.config.provider,
        model=model,
        reply=reply,
        tool_calls=tool_calls_log,
        usage=usage,
    )
