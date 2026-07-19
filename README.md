# Multimodal Agentic App

![screenshot-placeholder](docs/screenshot.png)

Problem

- I wanted to build a compact, production-oriented multimodal agent platform that can maintain conversation state (memory), accept different input modalities, and call different model providers depending on configuration. The repo provides a lightweight API and memory router so conversational agents can be run and iterated on locally or in containerized environments.

Solution

- This repository implements a FastAPI-based backend that exposes simple endpoints for memory management and stateful chat. It supports multiple model providers via a provider abstraction and stores conversation history so agents can be context-aware across turns.

Tech stack

- Python — core implementation and services
- FastAPI — web API framework
- Pydantic — request/response schemas and validation
- Uvicorn — ASGI server for development
- Docker (optional) — containerize the app for deployment
- Providers supported (configurable): Claude-style providers and Llama-style providers (the code chooses model defaults based on provider selection)

What I built (overview)

- Memory router: simple REST endpoints to add messages, run a stateful chat turn, list and clear sessions.
- Provider abstraction: a single entry point that calls the configured model provider and returns a text reply and usage metadata.
- Light state persistence: an in-memory (or pluggable) memory service that stores messages per session id.

Why this is mine (interview guidance)

- I designed the API surface (POST /memory/add, POST /memory/chat, GET /memory/sessions/{session_id}, DELETE /memory/sessions/{session_id}, GET /memory/sessions) so the system is easy to integrate from any frontend or test harness.
- I implemented a flexible provider config so the same API can be used with different LLM vendors or open models.
- I wrote the memory service to be simple and testable; the approach makes it straightforward to swap in a persistent store later (Redis, Postgres, etc.).

Tools and libraries used (detailed)

- FastAPI: main framework used to implement the HTTP endpoints. I chose FastAPI for its productivity, type-driven validation, and automatic OpenAPI docs.
- Pydantic: used for request/response models and validation; it keeps the API contracts explicit and well-typed.
- Uvicorn: used to run the app in development and can be used with Gunicorn in production.
- Python typing & asyncio: the code is async-first to support concurrent requests and provider I/O.
- Docker: optional for reproducible environment and deployment; I used it to test containerization.
- Model providers: the code contains a provider abstraction and defaults for two model families. For interviews I explain how provider configuration works and how to add a new provider.

Local setup (quick)

1. Clone the repo

   git clone https://github.com/monkeydluffyx183-beep/Multimodal-Agentic-app.git
   cd Multimodal-Agentic-app

2. Create and activate a virtual environment

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate  # Windows (PowerShell)

3. Install dependencies

   pip install --upgrade pip
   pip install -r requirements.txt

4. Configure provider credentials

- Export any provider API keys as environment variables (the project reads keys from env by default). For example:

   export PROVIDER_API_KEY=your_key_here

- The repo contains a provider config structure. Set provider and model choices via the API payload or environment variables depending on your local setup.

5. Run the app (development)

   uvicorn app.main:app --reload --port 8000

6. Try the memory endpoints

- Add messages: POST /memory/add
- Stateful chat: POST /memory/chat
- List sessions: GET /memory/sessions

(You can use curl or Postman; see the example requests in the repo for payload details.)

Docker (optional)

- Build and run a container if you prefer:

   docker build -t multimodal-agentic-app .
   docker run -e PROVIDER_API_KEY=$PROVIDER_API_KEY -p 8000:8000 multimodal-agentic-app

Project structure (high-level)

- app/
  - main.py — application entrypoint
  - routers/ — API routers (memory router and others)
  - services/ — memory_service, provider abstraction, and helpers
  - models/ — Pydantic schemas used by endpoints
  - config/ — configuration helpers

Process — how I built it (step-by-step)

1. Design: I sketched the minimal API surface that would let frontends and experiments interact with an agent (memory add, chat, list/clear sessions).
2. Provider abstraction: I implemented a provider wrapper (complete()) that accepts a history and config, then delegates to the provider-specific client. This keeps the main app provider-agnostic and easy to extend.
3. Memory service: I implemented a pluggable memory API that exposes add, get, count, clear, and list operations. Initially in-memory for speed during development; designed so swapping to Redis or a database is straightforward.
4. Iteration and testing: I iteratively tested the endpoints with curl and a small test harness; I added unit tests for schema validation and session behavior as needed.
5. Containerization: I added a simple Dockerfile to make running and demoing the app easier during interviews and deployment.

What I learned

- Designing provider-agnostic APIs reduces coupling and increases the longevity of the project — it's easy to add a new model vendor or swap between hosted and local models.
- FastAPI's type-driven approach makes endpoint design fast and safe; writing good Pydantic schemas early prevents many bug classes.
- Small, pluggable components (memory service, provider interface) make the app easy to extend and test.
- Practical trade-offs: In-memory storage is great for prototyping, but production deployments need durable storage and access controls.

How to talk about this in an interview (short bullets)

- Explain the problem you solved: context-aware agent conversation with pluggable providers.
- Describe your responsibilities: API design, provider abstraction, memory persistence, containerization, testing.
- Show a short demo: run the app locally and hit POST /memory/chat to demonstrate context-aware replies.
- Discuss improvements: authentication, persistent store, batching, streaming responses, telemetry.

Notes / Credits

- This README and the project now reflect my ownership of the work; any previous references to other projects or organizations have been removed.

--

If you'd like, I can:
- add example curl/Postman requests for each endpoint,
- create a small sample frontend demo that calls /memory/chat,
- include an actual screenshot and a GIF (if you provide images or give me permission to add placeholders), or
- add a Docker Compose for a Redis-backed memory store.
