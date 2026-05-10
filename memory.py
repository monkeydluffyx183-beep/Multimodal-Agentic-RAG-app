"""
Multi-Agent Router
POST /multi-agent/run   — run a team of agents on a task
GET  /multi-agent/presets — list built-in agent team presets
"""

from __future__ import annotations
import asyncio
from fastapi import APIRouter

from app.models.schemas import (
    MultiAgentRequest, MultiAgentResponse,
    AgentTurn, Message, AgentSpec, ProviderConfig,
)
from app.services.provider import complete

router = APIRouter()


# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS = {
    "finance_team": {
        "description": "Portfolio analysis — Analyst + Risk Manager + Strategist",
        "agents": [
            {"name": "Analyst",      "role": "Financial Analyst",  "system_prompt": "You are a sharp financial analyst. Analyse data, identify trends, and produce concise insights."},
            {"name": "Risk Manager", "role": "Risk Manager",        "system_prompt": "You assess financial risk. Review the analyst's findings and flag key risks."},
            {"name": "Strategist",   "role": "Investment Strategist","system_prompt": "You synthesise analysis and risk into a clear investment recommendation."},
        ],
    },
    "research_team": {
        "description": "Deep research — Researcher + Critic + Writer",
        "agents": [
            {"name": "Researcher", "role": "Researcher",    "system_prompt": "You gather and organise facts on a topic. Be thorough and cite specifics."},
            {"name": "Critic",     "role": "Critic",         "system_prompt": "You critically evaluate research. Identify gaps, biases, and unsupported claims."},
            {"name": "Writer",     "role": "Report Writer",  "system_prompt": "You write a clear, well-structured report based on research and critique."},
        ],
    },
    "legal_team": {
        "description": "Legal review — Paralegal + Lawyer + Summariser",
        "agents": [
            {"name": "Paralegal",   "role": "Paralegal",          "system_prompt": "You identify relevant legal issues and applicable laws."},
            {"name": "Lawyer",      "role": "Senior Lawyer",       "system_prompt": "You provide a detailed legal analysis and opinion."},
            {"name": "Summariser",  "role": "Executive Summariser","system_prompt": "You produce a concise executive summary of the legal findings."},
        ],
    },
}


@router.get("/presets")
async def list_presets():
    return {
        k: {"description": v["description"], "agents": [a["name"] for a in v["agents"]]}
        for k, v in PRESETS.items()
    }


# ── Orchestration strategies ──────────────────────────────────────────────────

async def _run_sequential(
    task: str,
    agents: list[AgentSpec],
    max_rounds: int,
) -> list[AgentTurn]:
    """Each agent receives the prior agent's output as context."""
    turns: list[AgentTurn] = []
    context = task

    for agent in agents:
        msg = Message(role="user", content=context)
        system = agent.system_prompt
        reply, _ = await complete([msg], agent.config, system)
        turns.append(AgentTurn(agent=agent.name, role=agent.role, output=reply))
        context = (
            f"Previous task: {task}\n\n"
            f"{agent.name} ({agent.role}) output:\n{reply}\n\n"
            "Continue based on the above."
        )

    return turns


async def _run_parallel(
    task: str,
    agents: list[AgentSpec],
    max_rounds: int,
) -> list[AgentTurn]:
    """All agents work on the task simultaneously."""
    async def _call(agent: AgentSpec) -> AgentTurn:
        reply, _ = await complete(
            [Message(role="user", content=task)],
            agent.config,
            agent.system_prompt,
        )
        return AgentTurn(agent=agent.name, role=agent.role, output=reply)

    results = await asyncio.gather(*[_call(a) for a in agents])
    return list(results)


async def _run_hierarchical(
    task: str,
    agents: list[AgentSpec],
    max_rounds: int,
) -> list[AgentTurn]:
    """
    First agent = orchestrator who delegates subtasks.
    Remaining agents = workers. Orchestrator produces final synthesis.
    """
    if len(agents) < 2:
        return await _run_sequential(task, agents, max_rounds)

    orchestrator = agents[0]
    workers = agents[1:]
    turns: list[AgentTurn] = []

    # Orchestrator breaks the task
    delegation_prompt = (
        f"You are managing a team: {', '.join(a.name for a in workers)}.\n"
        f"Task: {task}\n\n"
        f"Write a brief subtask for each team member separated by '---'. "
        f"Address each block with 'TO {worker.name.upper()}:' on the first line."
    )
    delegation, _ = await complete(
        [Message(role="user", content=delegation_prompt)],
        orchestrator.config,
        orchestrator.system_prompt,
    )
    turns.append(AgentTurn(
        agent=orchestrator.name,
        role=orchestrator.role,
        output=f"[Delegation]\n{delegation}",
    ))

    # Workers execute in parallel
    worker_results: list[AgentTurn] = []
    async def _worker_call(w: AgentSpec) -> AgentTurn:
        # Extract the relevant section
        section = delegation
        marker = f"TO {w.name.upper()}:"
        if marker in delegation:
            parts = delegation.split(marker, 1)
            section = parts[1].split("---")[0].strip()
        reply, _ = await complete(
            [Message(role="user", content=section)],
            w.config,
            w.system_prompt,
        )
        return AgentTurn(agent=w.name, role=w.role, output=reply)

    worker_outputs = await asyncio.gather(*[_worker_call(w) for w in workers])
    turns.extend(worker_outputs)

    # Orchestrator synthesises
    synthesis_ctx = "\n\n".join(
        f"### {t.agent} ({t.role})\n{t.output}" for t in worker_outputs
    )
    synthesis, _ = await complete(
        [Message(role="user", content=f"Synthesise these outputs into a final answer for: {task}\n\n{synthesis_ctx}")],
        orchestrator.config,
        orchestrator.system_prompt,
    )
    turns.append(AgentTurn(
        agent=orchestrator.name,
        role=orchestrator.role,
        output=f"[Synthesis]\n{synthesis}",
    ))
    return turns


# ── Main endpoint ──────────────────────────────────────────────────────────────

@router.post("/run", response_model=MultiAgentResponse)
async def run_multi_agent(req: MultiAgentRequest):
    # Ensure each AgentSpec has a config
    for spec in req.agents:
        if spec.config is None:
            spec.config = ProviderConfig()

    if req.orchestration == "sequential":
        turns = await _run_sequential(req.task, req.agents, req.max_rounds)
    elif req.orchestration == "parallel":
        turns = await _run_parallel(req.task, req.agents, req.max_rounds)
    else:
        turns = await _run_hierarchical(req.task, req.agents, req.max_rounds)

    # Summarise
    summary_ctx = "\n".join(f"[{t.agent}]: {t.output[:300]}" for t in turns)
    summary_cfg = req.agents[-1].config if req.agents else ProviderConfig()
    summary, _ = await complete(
        [Message(role="user", content=f"Summarise the key outcome in 2 sentences:\n{summary_ctx}")],
        summary_cfg,
    )

    return MultiAgentResponse(
        task=req.task,
        orchestration=req.orchestration,
        turns=turns,
        final_summary=summary,
        total_tokens=0,   # extend: sum usage across calls
    )
