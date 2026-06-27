# ATLAS — Self-Healing Multi-Agent Code Assistant

> The AI backend powering DesignPro. Agents that write, test, and fix their own code.

```
┌──────────────────────────────────────────┐
│  FastAPI + WebSockets (this repo)         │
├──────────────────────────────────────────┤
│  L4 GOVERNOR   — model cost routing      │
│  L3 ORCHESTRATOR — LangGraph DAG         │
│  L2 ARCHITECT | CODER — specialists      │
│  L1 VERIFIER   — Docker test gate        │
│  L0 TOOLS      — FAISS | Postgres        │
└──────────────────────────────────────────┘
```

**Core invariant**: Nothing propagates upward until it passes L1 verification.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | [python.org](https://www.python.org) |
| Poetry | 1.8+ | `pip install poetry` |
| Docker | 24+ | [docker.com](https://www.docker.com) |
| Git | any | [git-scm.com](https://git-scm.com) |

---

## Phase 0 — Local Setup (Start Here)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/atlas.git
cd atlas

# Install all dependencies
poetry install

# Activate the virtualenv
poetry shell
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:
```env
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

You don't need Postgres, Docker, or LangSmith for Phase 0.

### 3. Start the API

```bash
python main.py
```

You should see:
```
INFO  atlas_starting version=0.1.0 environment=development
INFO  atlas_ready host=0.0.0.0 port=8000
```

### 4. Test it

```bash
# Health check
curl http://localhost:8000/health

# Create a task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Write a fibonacci function",
    "description": "Write a Python function that computes the nth Fibonacci number using dynamic programming with memoization",
    "language": "python"
  }'

# Interactive docs
open http://localhost:8000/docs
```

### 5. Run tests

```bash
pytest
# or with verbose output
pytest -v
```

All tests should pass without any API keys (no LLM calls in Phase 0 tests).

---

## Phase 1 — Critic Loop ⭐

Coming next. The CoderAgent writes code, the VerificationGate runs it in Docker,
and the loop retries until tests pass.

**What you'll build:**
- `atlas/agents/coder.py` — calls Anthropic, produces code artifacts
- `atlas/tools/sandbox.py` — Docker subprocess isolation
- `atlas/agents/verifier.py` — runs pytest inside the sandbox

---

## Phase 2 — Orchestration

**What you'll build:**
- `atlas/agents/architect.py` — decomposes tasks into subtasks
- `atlas/orchestrator.py` — LangGraph DAG connecting all agents
- PostgreSQL migration — tasks persist across restarts

---

## Phase 3 — Memory + Governor

**What you'll build:**
- `atlas/memory/failure_store.py` — FAISS index of past failures
- `atlas/governor/router.py` — cost-aware model selection

---

## Phase 4 — Frontend

**What you'll build:**
- DesignPro landing page (React + Framer Motion)
- ATLAS dashboard (agent feed + DAG viz + cost panel)

---

## Phase 5 — Deploy

### Railway (Backend)

1. Create a Railway account at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Add environment variables (copy from `.env.example`)
4. Railway auto-detects the Dockerfile

```bash
# Or deploy from CLI
npm install -g @railway/cli
railway login
railway up
```

### Vercel (Frontend — Phase 4)

```bash
cd designpro
npm install -g vercel
vercel --prod
```

### GitHub Actions (CI/CD)

Add these secrets in GitHub → Settings → Secrets → Actions:
- `RAILWAY_TOKEN` — from Railway dashboard → Settings → Tokens
- `RAILWAY_SERVICE` — your service ID from the Railway URL

Every push to `main` runs tests then deploys automatically.

---

## Project Structure

```
atlas/
├── atlas/
│   ├── contracts.py          ← Pydantic v2 type contracts (source of truth)
│   ├── config.py             ← Settings from environment variables
│   ├── logging.py            ← Structured logging (JSON in prod, pretty in dev)
│   ├── agents/
│   │   ├── base.py           ← BaseAgent (retry, logging, cost tracking)
│   │   ├── architect.py      ← Phase 2: task decomposition
│   │   ├── coder.py          ← Phase 1: code generation
│   │   └── verifier.py       ← Phase 1: test execution
│   ├── api/
│   │   ├── app.py            ← FastAPI application + all routes
│   │   ├── task_service.py   ← Business logic (swap storage here)
│   │   └── websocket_manager.py  ← Real-time event broadcasting
│   ├── tools/
│   │   └── sandbox.py        ← Phase 1: Docker subprocess runner
│   ├── memory/
│   │   └── failure_store.py  ← Phase 3: FAISS failure index
│   └── governor/
│       └── router.py         ← Phase 3: cost-aware model routing
├── tests/
│   ├── unit/                 ← Fast tests, no I/O
│   └── integration/          ← Tests with real DB / Docker (Phase 1+)
├── main.py                   ← uvicorn entrypoint
├── Dockerfile                ← Multi-stage production build
├── docker-compose.yml        ← Local dev with Postgres
├── pyproject.toml            ← Poetry dependencies
└── .github/workflows/ci.yml  ← GitHub Actions
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/tasks` | Submit a task (202 Accepted) |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/{id}` | Get task detail |
| `DELETE` | `/tasks/{id}` | Cancel a task |
| `WS` | `/ws/{task_id}` | Subscribe to live events |

Interactive docs: `http://localhost:8000/docs` (dev only)

> **Dev note — hot-reload and in-flight tasks**: `main.py` starts uvicorn with
> `reload=True` in non-production environments. Any file change in the project
> directory triggers a full server restart, which `asyncio.CancelledError`s every
> running pipeline. Tasks left mid-flight will stay in their last status (e.g.
> `coding`) with no cleanup. Either set `ENVIRONMENT=production` to disable reload,
> or use `--reload-exclude` to exclude non-source directories (e.g. test output,
> generated files). This is expected uvicorn behaviour, not a bug.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + WebSockets |
| Contracts | Pydantic v2 |
| LLMs | Anthropic + OpenAI |
| Orchestration | LangGraph |
| Memory/RAG | FAISS + sentence-transformers |
| State | PostgreSQL (event sourcing) |
| Sandbox | Docker subprocess |
| Observability | structlog + LangSmith |
| CI/CD | GitHub Actions |
| Deploy | Railway (API) + Vercel (frontend) |
