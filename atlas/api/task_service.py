"""
atlas/api/task_service.py

TaskService — the business logic layer between the API and the agents.

Phase 0: Pure in-memory dict. No database dependency.
Phase 2: Swap the dict for asyncpg + SQLAlchemy — the API layer doesn't change.

This separation (API ↔ Service ↔ Storage) means we can test the service
without starting a database, and swap storage without touching routes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from atlas.contracts import (
    Artifact,
    SubTask,
    TaskDetail,
    TaskRequest,
    TaskResponse,
    TaskStatus,
)
from atlas.logging import get_logger

log = get_logger(__name__)


class TaskNotFoundError(Exception):
    """Raised when a task_id doesn't exist in the store."""
    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task not found: {task_id}")
        self.task_id = task_id


class TaskService:
    """
    Manages task lifecycle for ATLAS.

    Thread-safety note: FastAPI runs in a single-process async event loop.
    A plain dict is safe here because Python's GIL serialises dict operations,
    and we're not running multiple processes. For multi-process deployments
    (Phase 5 with Railway scaling), swap to Redis or Postgres.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskDetail] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, request: TaskRequest) -> TaskResponse:
        """Accept a new task and store it as PENDING."""
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        task = TaskDetail(
            task_id=task_id,
            title=request.title,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        self._tasks[task_id] = task

        log.info(
            "task_created",
            task_id=task_id,
            title=request.title,
            language=request.language,
        )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=now,
        )

    async def get(self, task_id: str) -> TaskDetail:
        """Fetch a task by ID. Raises TaskNotFoundError if missing."""
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        return self._tasks[task_id]

    async def list_all(self, limit: int = 50) -> list[TaskDetail]:
        """Return the N most recent tasks, newest first."""
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )
        return tasks[:limit]

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: str | None = None,
    ) -> TaskDetail:
        """Transition a task to a new status."""
        task = await self.get(task_id)

        # Pydantic frozen models — we must rebuild with changed fields
        updated = task.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(timezone.utc),
                "error": error,
            }
        )
        self._tasks[task_id] = updated

        log.info("task_status_changed", task_id=task_id, status=status)
        return updated

    async def add_subtasks(
        self, task_id: str, subtasks: list[SubTask]
    ) -> TaskDetail:
        """Attach ArchitectAgent's subtask plan to a task."""
        task = await self.get(task_id)
        updated = task.model_copy(
            update={
                "subtasks": subtasks,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._tasks[task_id] = updated
        return updated

    async def add_artifact(self, task_id: str, artifact: Artifact) -> TaskDetail:
        """Append a produced file to a task's artifacts."""
        task = await self.get(task_id)
        updated = task.model_copy(
            update={
                "artifacts": [*task.artifacts, artifact],
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._tasks[task_id] = updated
        return updated

    async def increment_cost(self, task_id: str, cost_usd: float) -> None:
        """Add to the running cost total for a task."""
        task = await self.get(task_id)
        updated = task.model_copy(
            update={
                "cost_usd": task.cost_usd + cost_usd,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._tasks[task_id] = updated

    # ------------------------------------------------------------------
    # Stats (used by the dashboard)
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Quick summary for the /health endpoint and dashboard."""
        total = len(self._tasks)
        by_status: dict[str, int] = {}
        total_cost = 0.0

        for task in self._tasks.values():
            by_status[task.status] = by_status.get(task.status, 0) + 1
            total_cost += task.cost_usd

        return {
            "total_tasks": total,
            "by_status": by_status,
            "total_cost_usd": round(total_cost, 4),
        }
