"""
atlas/api/app.py — Phase 2 updated.

New endpoints:
  GET /tasks/{task_id}/timeline  → full event history (DAG visualiser feed)
  GET /tasks/{task_id}/plan      → the execution plan with DAG structure
  GET /stats                     → system-wide stats

Pipeline replaces CriticLoop as the entry point.
CriticLoop still runs internally for simple (score 1-3) tasks.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from atlas.api.task_service import TaskNotFoundError, TaskService
from atlas.api.websocket_manager import ws_manager
from atlas.config import settings
from atlas.contracts import (
    HealthResponse,
    TaskDetail,
    TaskRequest,
    TaskResponse,
)
from atlas.event_store import event_store
from atlas.governor.governor import governor
from atlas.memory.failure_memory import failure_memory
from atlas.pipeline import Pipeline
from atlas.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    log.info(
        "atlas_starting",
        version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
    )
    app.state.task_service = TaskService()

    # Initialize Phase 3 singletons
    failure_memory.initialize()
    log.info(
        "atlas_ready",
        host=settings.host,
        port=settings.port,
        memory_patterns=failure_memory.pattern_count,
        memory_available=failure_memory.is_available,
    )
    yield
    log.info("atlas_shutdown", active_ws=ws_manager.total_connections)


def create_app() -> FastAPI:
    # Keyed by task_id — lets DELETE cancel a running pipeline (A2).
    _pipeline_tasks: dict[str, asyncio.Task] = {}
    # Cap concurrent pipelines to settings.max_concurrent_tasks (B4).
    _pipeline_semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)

    app = FastAPI(
        title="ATLAS API",
        description="Self-healing multi-agent code assistant — Phase 2",
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(TaskNotFoundError)
    async def task_not_found_handler(request: Request, exc: TaskNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc), "task_id": exc.task_id},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        log.error("unhandled_exception", error=str(exc), error_type=type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ------------------------------------------------------------------
    # System endpoints
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health(request: Request) -> HealthResponse:
        svc: TaskService = request.app.state.task_service
        stats = svc.stats()
        return HealthResponse(
            status="ok",
            version=settings.app_version,
            services={
                "tasks": str(stats["total_tasks"]),
                "websocket": f"{ws_manager.total_connections} connections",
                "events": str(sum(
                    event_store.count(tid)
                    for tid in [t.task_id for t in (await svc.list_all())]
                )),
                **{k: str(v) for k, v in stats["by_status"].items()},
            },
        )

    @app.get("/stats", tags=["System"])
    async def system_stats(request: Request) -> dict:
        """System-wide statistics for the dashboard."""
        svc: TaskService = request.app.state.task_service
        all_budgets = governor.get_all_budgets()
        return {
            "tasks": svc.stats(),
            "websocket": {
                "active_connections": ws_manager.total_connections,
                "active_tasks": ws_manager.active_tasks,
            },
            "memory": {
                "patterns_stored": failure_memory.pattern_count,
                "available": failure_memory.is_available,
            },
            "governor": {
                "active_tasks": len(all_budgets),
                "total_spent_usd": round(sum(b.spent_usd for b in all_budgets.values()), 6),
                "total_llm_calls": sum(b.llm_calls for b in all_budgets.values()),
                "total_tokens": sum(b.tokens_in + b.tokens_out for b in all_budgets.values()),
                "tier_downgrades": sum(b.tier_downgrades for b in all_budgets.values()),
            },
        }

    @app.get("/tasks/{task_id}/budget", tags=["Tasks"])
    async def get_budget(request: Request, task_id: str) -> dict:
        """Real-time budget state for a task."""
        svc: TaskService = request.app.state.task_service
        await svc.get(task_id)
        budget = governor.get_budget(task_id)
        if not budget:
            return {"task_id": task_id, "message": "No budget data yet"}
        return budget.model_dump()

    @app.get("/memory/stats", tags=["System"])
    async def memory_stats() -> dict:
        """Failure memory statistics."""
        return {
            "patterns_stored": failure_memory.pattern_count,
            "available": failure_memory.is_available,
            "message": "Memory grows with every retry — the system learns from bugs",
        }

    # ------------------------------------------------------------------
    # Task endpoints
    # ------------------------------------------------------------------

    @app.post("/tasks", response_model=TaskResponse, status_code=202, tags=["Tasks"])
    async def create_task(request: Request, body: TaskRequest) -> TaskResponse:
        """
        Submit a task to ATLAS Phase 2 Pipeline.

        The pipeline will:
          1. Score complexity (IntelligenceRouter)
          2. Decompose into DAG (ArchitectAgent) if complex
          3. Execute waves in parallel (Orchestrator)
          4. Self-heal on failures (CriticLoop per subtask)
        """
        svc: TaskService = request.app.state.task_service
        response = await svc.create(body)
        task_id = response.task_id
        log.info("api_task_created", task_id=task_id)

        pipeline = Pipeline(task_id=task_id, task_service=svc)

        async def _guarded() -> None:
            async with _pipeline_semaphore:   # B4: cap concurrency
                await pipeline.run(body)
            _pipeline_tasks.pop(task_id, None)

        handle = asyncio.create_task(_guarded())
        _pipeline_tasks[task_id] = handle     # A2: store for cancellation
        return response

    @app.get("/tasks", response_model=list[TaskDetail], tags=["Tasks"])
    async def list_tasks(request: Request, limit: int = 50) -> list[TaskDetail]:
        svc: TaskService = request.app.state.task_service
        return await svc.list_all(limit=limit)

    @app.get("/tasks/{task_id}", response_model=TaskDetail, tags=["Tasks"])
    async def get_task(request: Request, task_id: str) -> TaskDetail:
        svc: TaskService = request.app.state.task_service
        return await svc.get(task_id)

    @app.get("/tasks/{task_id}/timeline", tags=["Tasks"])
    async def get_timeline(request: Request, task_id: str) -> dict:
        """
        Full event history for a task — powers the DAG visualiser.

        Every agent call, every test result, every state transition.
        The frontend renders this as a live-updating timeline graph.
        """
        svc: TaskService = request.app.state.task_service
        task = await svc.get(task_id)  # raises 404 if not found
        events = event_store.timeline(task_id)
        stats = event_store.stats(task_id)
        return {
            "task_id": task_id,
            "title": task.title,
            "status": task.status,
            "events": events,
            "stats": stats,
            "event_count": len(events),
        }

    @app.delete("/tasks/{task_id}", status_code=204, tags=["Tasks"])
    async def cancel_task(request: Request, task_id: str) -> Response:
        svc: TaskService = request.app.state.task_service
        task = await svc.get(task_id)
        if task.status in ("completed", "failed"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task in status '{task.status}'",
            )
        # A2: cancel the running asyncio task so the pipeline actually stops
        handle = _pipeline_tasks.pop(task_id, None)
        if handle and not handle.done():
            handle.cancel()
        await svc.update_status(task_id, "failed", error="Cancelled by user")
        return Response(status_code=204)

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws/{task_id}")
    async def websocket_endpoint(websocket: WebSocket, task_id: str) -> None:
        """
        Subscribe to live events for a task.

        Receives all events: agent.started, agent.completed, test.result,
        task.updated (with plan + wave info), stream.token, etc.

        Connect from frontend:
            const ws = new WebSocket(`ws://localhost:8000/ws/${taskId}`);
            ws.onmessage = e => {
              const event = JSON.parse(e.data);
              console.log(event.event, event.data);
            };
        """
        await ws_manager.connect(websocket, task_id)
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, task_id)

    return app


app = create_app()
