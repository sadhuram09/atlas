"""
atlas/contracts_v2.py

Phase 2 extensions to the contract layer.

New concepts:
  - ExecutionPlan: the dependency graph ArchitectAgent produces
  - SubTaskNode: a subtask with dependency edges (DAG node)
  - ExecutionEvent: immutable event for the event store
  - ComplexityScore: Intelligence Router output
  - PipelineResult: final result of the full orchestration

These extend contracts.py without modifying it — backward compatible.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atlas.contracts import ATLASModel, AgentRole, Artifact, ModelTier, TestResult


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

class ComplexityScore(ATLASModel):
    """
    Output of the IntelligenceRouter — scores a task before work begins.

    Score 1-3  → FAST model (llama-3.1-8b-instant)
    Score 4-7  → BALANCED model (llama-3.3-70b-versatile)
    Score 8-10 → POWERFUL model (deepseek-r1-distill-llama-70b)
    """
    score: int = Field(ge=1, le=10)
    tier: ModelTier
    reasoning: str
    estimated_subtasks: int = Field(ge=1, le=20)
    requires_parallel: bool = False


# ---------------------------------------------------------------------------
# DAG — Directed Acyclic Graph of subtasks
# ---------------------------------------------------------------------------

class SubTaskStatus(StrEnum):
    WAITING   = "waiting"    # Blocked by dependencies
    READY     = "ready"      # All deps done, can execute
    RUNNING   = "running"    # Currently executing
    PASSED    = "passed"     # Verified and complete
    FAILED    = "failed"     # Exhausted retries


class SubTaskNode(ATLASModel):
    """
    A node in the execution DAG.

    depends_on: list of subtask IDs that must PASS before this runs.
    An empty depends_on means this subtask is immediately ready.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)  # IDs of blocking subtasks
    order: int = Field(ge=0)
    status: SubTaskStatus = SubTaskStatus.WAITING
    artifacts: list[Artifact] = Field(default_factory=list)
    test_result: TestResult | None = None
    attempts: int = 0
    assigned_tier: ModelTier = ModelTier.BALANCED


class ExecutionPlan(ATLASModel):
    """
    The full DAG produced by ArchitectAgent.

    nodes: all subtasks
    edges: dependency relationships (from_id must complete before to_id)
    execution_waves: pre-computed parallel batches
      Wave 0: all nodes with no dependencies (run in parallel)
      Wave 1: all nodes whose deps are all in wave 0 (run in parallel)
      etc.
    """
    task_id: str
    nodes: list[SubTaskNode]
    edges: list[tuple[str, str]] = Field(default_factory=list)  # (from_id, to_id)
    execution_waves: list[list[str]] = Field(default_factory=list)  # wave → [node_ids]
    complexity: ComplexityScore
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_node(self, node_id: str) -> SubTaskNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def ready_nodes(self) -> list[SubTaskNode]:
        """Return all nodes that are READY to execute right now."""
        return [n for n in self.nodes if n.status == SubTaskStatus.READY]

    def all_passed(self) -> bool:
        return all(n.status == SubTaskStatus.PASSED for n in self.nodes)

    def any_failed(self) -> bool:
        return any(n.status == SubTaskStatus.FAILED for n in self.nodes)


# ---------------------------------------------------------------------------
# Event sourcing
# ---------------------------------------------------------------------------

class EventKind(StrEnum):
    """Every possible thing that can happen in the system."""
    # Task lifecycle
    TASK_ACCEPTED          = "task.accepted"
    TASK_COMPLEXITY_SCORED = "task.complexity_scored"
    TASK_PLAN_CREATED      = "task.plan_created"
    TASK_COMPLETED         = "task.completed"
    TASK_FAILED            = "task.failed"

    # Subtask lifecycle
    SUBTASK_READY          = "subtask.ready"
    SUBTASK_STARTED        = "subtask.started"
    SUBTASK_CODED          = "subtask.coded"
    SUBTASK_VERIFIED       = "subtask.verified"
    SUBTASK_PASSED         = "subtask.passed"
    SUBTASK_FAILED         = "subtask.failed"
    SUBTASK_RETRY          = "subtask.retry"

    # Agent events
    AGENT_CALLED           = "agent.called"
    AGENT_RESPONDED        = "agent.responded"
    LLM_CALL_MADE          = "llm.call_made"

    # Parallel execution
    WAVE_STARTED           = "wave.started"
    WAVE_COMPLETED         = "wave.completed"


class ExecutionEvent(ATLASModel):
    """
    An immutable event in the event store.

    Every state change in ATLAS is recorded as an event.
    The sequence of events for a task_id is its complete history.
    You can replay events to reconstruct any past state.

    This is event sourcing — the same pattern used by financial systems
    and distributed databases because it gives you:
      - Complete audit trail
      - Time-travel debugging (replay to any point)
      - The live DAG visualiser just subscribes to this stream
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    kind: EventKind
    agent: AgentRole | None = None
    subtask_id: str | None = None
    wave: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int | None = None


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

class PipelineResult(ATLASModel):
    """Final result after all waves complete."""
    task_id: str
    success: bool
    total_subtasks: int
    passed_subtasks: int
    failed_subtasks: int
    total_attempts: int
    all_artifacts: list[Artifact] = Field(default_factory=list)
    waves_executed: int = 0
    total_duration_ms: int = 0
    complexity_score: int = 0
