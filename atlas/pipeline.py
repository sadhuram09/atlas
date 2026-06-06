"""
atlas/pipeline.py

The full Phase 2 pipeline — replaces the Phase 1 CriticLoop for complex tasks.

Flow:
  POST /tasks
      ↓
  Pipeline.run()
      ↓
  [1] IntelligenceRouter  → scores complexity, picks model tiers
      ↓
  [2] ArchitectAgent       → decomposes into DAG with dependency edges
      ↓
  [3] Orchestrator          → executes waves in parallel
        Wave 0: [subtask_A, subtask_B]  ← run simultaneously
        Wave 1: [subtask_C]             ← runs after A + B pass
        Wave 2: [subtask_D]             ← runs after C passes
      ↓
  [4] PipelineResult        → task marked COMPLETED or FAILED

Smart routing:
  - complexity.score 1-3 → skip ArchitectAgent, single subtask, use CriticLoop
  - complexity.score 4+  → full DAG decomposition + parallel orchestration

This means simple tasks ("write an add function") stay fast and lean.
Complex tasks ("build a URL shortener") get the full power.

Every step emits events to the event store AND WebSocket.
"""

from __future__ import annotations

import asyncio

from atlas.agents.architect import ArchitectAgent
from atlas.agents.intelligence_router import IntelligenceRouter
from atlas.api.task_service import TaskService
from atlas.api.websocket_manager import ws_manager
from atlas.contracts import (
    AgentMessage,
    AgentRole,
    EventType,
    StreamEvent,
    TaskRequest,
    TaskStatus,
)
from atlas.contracts_v2 import (
    ComplexityScore,
    EventKind,
    ExecutionPlan,
    SubTaskNode,
    SubTaskStatus,
)
from atlas.critic_loop import CriticLoop
from atlas.event_store import event_store
from atlas.governor.governor import governor
from atlas.memory.failure_memory import failure_memory
from atlas.orchestrator import Orchestrator
from atlas.logging import get_logger

log = get_logger(__name__)


class Pipeline:
    """
    Full Phase 2 pipeline: route → plan → execute.

    One instance per task. Called as an asyncio background task.
    """

    def __init__(self, task_id: str, task_service: TaskService) -> None:
        self.task_id = task_id
        self.svc = task_service
        self.log = get_logger(__name__).bind(task_id=task_id)

    async def run(self, request: TaskRequest) -> None:
        """
        Execute the full pipeline for a task.

        Handles all errors internally — never raises.
        Always resolves the task to COMPLETED or FAILED.
        """
        self.log.info("pipeline_started", title=request.title)

        event_store.append(
            self.task_id,
            EventKind.TASK_ACCEPTED,
            payload={"title": request.title, "language": request.language},
        )

        try:
            await self._run_internal(request)
        except Exception as e:
            self.log.error("pipeline_crashed", error=str(e))
            await self._fail(f"Pipeline crashed: {e}")

    async def _run_internal(self, request: TaskRequest) -> None:

        # ── STEP 1: INTELLIGENCE ROUTING ─────────────────────────────
        await self.svc.update_status(self.task_id, TaskStatus.PLANNING)
        await self._ws(EventType.TASK_UPDATED, {
            "status": TaskStatus.PLANNING,
            "step": "scoring_complexity",
        })

        complexity = await self._score_complexity(request)

        event_store.append(
            self.task_id,
            EventKind.TASK_COMPLEXITY_SCORED,
            payload={
                "score": complexity.score,
                "tier": complexity.tier,
                "reasoning": complexity.reasoning,
                "estimated_subtasks": complexity.estimated_subtasks,
                "requires_parallel": complexity.requires_parallel,
            },
        )

        await self._ws(EventType.TASK_UPDATED, {
            "status": TaskStatus.PLANNING,
            "step": "complexity_scored",
            "complexity_score": complexity.score,
            "tier": complexity.tier,
            "reasoning": complexity.reasoning,
            "estimated_subtasks": complexity.estimated_subtasks,
        })

        # ── STEP 2: SIMPLE TASK FAST PATH ─────────────────────────────
        # Score 1-3 with 1 subtask → skip ArchitectAgent overhead
        if complexity.score <= 3 and complexity.estimated_subtasks == 1:
            self.log.info(
                "pipeline_fast_path",
                score=complexity.score,
                reason="Simple task — using CriticLoop directly",
            )
            await self._ws(EventType.TASK_UPDATED, {
                "status": TaskStatus.CODING,
                "step": "fast_path",
                "reason": "Simple task — direct execution",
            })
            # Delegate to Phase 1 CriticLoop
            loop = CriticLoop(task_id=self.task_id, task_service=self.svc)
            await loop.run(request)
            return

        # ── STEP 3: ARCHITECT DECOMPOSITION ──────────────────────────
        await self._ws(EventType.TASK_UPDATED, {
            "status": TaskStatus.PLANNING,
            "step": "decomposing",
            "message": f"Decomposing into {complexity.estimated_subtasks} subtasks...",
        })

        plan = await self._decompose(request, complexity)

        event_store.append(
            self.task_id,
            EventKind.TASK_PLAN_CREATED,
            payload={
                "subtasks": len(plan.nodes),
                "waves": len(plan.execution_waves),
                "edges": len(plan.edges),
                "node_titles": [n.title for n in plan.nodes],
            },
        )

        await self._ws(EventType.TASK_UPDATED, {
            "status": TaskStatus.PLANNING,
            "step": "plan_created",
            "subtasks": len(plan.nodes),
            "waves": len(plan.execution_waves),
            "plan": {
                "nodes": [
                    {
                        "id": n.id,
                        "title": n.title,
                        "depends_on": n.depends_on,
                        "tier": n.assigned_tier,
                        "status": n.status,
                    }
                    for n in plan.nodes
                ],
                "waves": plan.execution_waves,
            },
        })

        # ── STEP 4: PARALLEL ORCHESTRATION ───────────────────────────
        await self.svc.update_status(self.task_id, TaskStatus.CODING)

        orchestrator = Orchestrator(
            task_id=self.task_id,
            task_service=self.svc,
        )
        result = await orchestrator.execute(plan, request)

        # ── STEP 5: FINAL VERDICT ─────────────────────────────────────
        if result.success:
            await self.svc.update_status(self.task_id, TaskStatus.COMPLETED)

            event_store.append(
                self.task_id,
                EventKind.TASK_COMPLETED,
                payload={
                    "subtasks": result.total_subtasks,
                    "attempts": result.total_attempts,
                    "duration_ms": result.total_duration_ms,
                    "artifacts": len(result.all_artifacts),
                },
            )

            await self._ws(EventType.TASK_UPDATED, {
                "status": TaskStatus.COMPLETED,
                "subtasks_completed": result.passed_subtasks,
                "total_subtasks": result.total_subtasks,
                "total_attempts": result.total_attempts,
                "duration_ms": result.total_duration_ms,
                "waves_executed": result.waves_executed,
                "complexity_score": result.complexity_score,
                "message": (
                    f"All {result.total_subtasks} subtasks completed "
                    f"across {result.waves_executed} waves "
                    f"in {result.total_duration_ms}ms"
                ),
            })

            self.log.info(
                "pipeline_completed",
                subtasks=result.total_subtasks,
                waves=result.waves_executed,
                duration_ms=result.total_duration_ms,
            )

        else:
            await self._fail(
                f"{result.failed_subtasks}/{result.total_subtasks} subtasks failed "
                f"after {result.total_attempts} total attempts"
            )

    # ------------------------------------------------------------------
    # Agent calls
    # ------------------------------------------------------------------

    async def _score_complexity(self, request: TaskRequest) -> ComplexityScore:
        """Call IntelligenceRouter and return a ComplexityScore."""
        msg = AgentMessage(
            task_id=self.task_id,
            from_agent=AgentRole.ORCHESTRATOR,
            to_agent=AgentRole.GOVERNOR,
            payload={
                "task_title": request.title,
                "task_description": request.description,
                "language": request.language,
            },
        )
        async with IntelligenceRouter(self.task_id) as router:
            result = await router.run(msg)

        return ComplexityScore(**result.payload["complexity"])

    async def _decompose(
        self, request: TaskRequest, complexity: ComplexityScore
    ) -> ExecutionPlan:
        """Call ArchitectAgent and return an ExecutionPlan."""
        msg = AgentMessage(
            task_id=self.task_id,
            from_agent=AgentRole.ORCHESTRATOR,
            to_agent=AgentRole.ARCHITECT,
            payload={
                "task_title": request.title,
                "task_description": request.description,
                "language": request.language,
                "complexity": complexity.model_dump(),
            },
        )
        async with ArchitectAgent(self.task_id) as architect:
            result = await architect.run(msg)

        return ExecutionPlan(**result.payload["plan"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ws(self, event_type: EventType, data: dict) -> None:
        await ws_manager.broadcast(
            StreamEvent(event=event_type, task_id=self.task_id, data=data)
        )

    async def _fail(self, reason: str) -> None:
        await self.svc.update_status(self.task_id, TaskStatus.FAILED, error=reason)
        event_store.append(
            self.task_id,
            EventKind.TASK_FAILED,
            payload={"reason": reason},
        )
        await self._ws(EventType.TASK_UPDATED, {
            "status": TaskStatus.FAILED,
            "error": reason,
        })
        self.log.error("pipeline_failed", reason=reason)
