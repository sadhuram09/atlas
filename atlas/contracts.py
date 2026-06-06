"""
atlas/contracts.py

Pydantic v2 contracts — the single source of truth for every message
that flows between agents, the API, and the frontend.

Design principle: if it crosses a boundary (HTTP, WebSocket, agent→agent),
it lives here. This makes the whole system auditable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Lifecycle of a task through the ATLAS pipeline."""
    PENDING      = "pending"       # Accepted, not yet started
    PLANNING     = "planning"      # ArchitectAgent decomposing the task
    CODING       = "coding"        # CoderAgent writing solution
    VERIFYING    = "verifying"     # VerificationGate running tests
    RETRY        = "retry"         # Failed verification — retrying
    COMPLETED    = "completed"     # Passed all gates
    FAILED       = "failed"        # Exhausted retries


class AgentRole(StrEnum):
    """Every agent type in the system."""
    GOVERNOR     = "governor"      # L4 — model routing + cost decisions
    ORCHESTRATOR = "orchestrator"  # L3 — LangGraph DAG coordinator
    ARCHITECT    = "architect"     # L2 — decomposes task into subtasks
    CODER        = "coder"         # L2 — writes code solutions
    VERIFIER     = "verifier"      # L1 — runs tests in Docker sandbox


class EventType(StrEnum):
    """Fine-grained events streamed to the frontend via WebSocket."""
    TASK_CREATED    = "task.created"
    TASK_UPDATED    = "task.updated"
    AGENT_STARTED   = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED    = "agent.failed"
    STEP_CREATED    = "step.created"
    STEP_RESULT     = "step.result"
    TEST_RESULT     = "test.result"
    STREAM_TOKEN    = "stream.token"    # LLM token streaming
    COST_UPDATE     = "cost.update"


class ModelTier(StrEnum):
    """Cost tiers — Governor picks the cheapest model that can solve the task."""
    FAST      = "fast"       # claude-haiku  — drafts, simple edits
    BALANCED  = "balanced"   # claude-sonnet — most tasks
    POWERFUL  = "powerful"   # claude-opus   — hard reasoning, final pass


# ---------------------------------------------------------------------------
# Base model config
# ---------------------------------------------------------------------------


class ATLASModel(BaseModel):
    """Base for all ATLAS contracts. Frozen = immutable after creation."""
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Task contracts
# ---------------------------------------------------------------------------


class TaskRequest(ATLASModel):
    """Inbound: what the caller wants done."""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10, max_length=4000)
    language: str = Field(default="python", description="Target programming language")
    max_retries: int = Field(default=3, ge=1, le=10)
    budget_usd: float = Field(default=0.50, gt=0, le=10.0, description="Max LLM cost")


class SubTask(ATLASModel):
    """ArchitectAgent output — one atomic unit of work for CoderAgent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    order: int = Field(ge=0)


class TaskResponse(ATLASModel):
    """API response after accepting a task."""
    task_id: str
    status: TaskStatus
    created_at: datetime
    estimated_cost_usd: float | None = None
    message: str = "Task accepted and queued"


class TaskDetail(ATLASModel):
    """Full task state — returned by GET /task/{id}."""
    task_id: str
    title: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    subtasks: list[SubTask] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    cost_usd: float = 0.0
    attempt: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Code artifacts
# ---------------------------------------------------------------------------


class Artifact(ATLASModel):
    """A file produced by the CoderAgent."""
    filename: str
    language: str
    content: str
    checksum: str = ""  # SHA-256 of content, populated by service layer


class TestResult(ATLASModel):
    """VerificationGate output — did the code pass?"""
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    test_count: int = 0
    failed_tests: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent message bus
# ---------------------------------------------------------------------------


class AgentMessage(ATLASModel):
    """
    Typed envelope for every inter-agent communication.

    Agents never call each other directly — they exchange AgentMessages.
    This makes the whole system auditable and replayable.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    from_agent: AgentRole
    to_agent: AgentRole
    payload: dict[str, Any]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# WebSocket event stream
# ---------------------------------------------------------------------------


class StreamEvent(ATLASModel):
    """
    Every event pushed to the frontend over WebSocket.

    The frontend subscribes once per task and receives a stream of these.
    Event type determines how the dashboard renders it.
    """
    event: EventType
    task_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    data: dict[str, Any] = Field(default_factory=dict)


class CostEvent(ATLASModel):
    """Emitted after every LLM call so the dashboard can track spend."""
    task_id: str
    agent: AgentRole
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(ATLASModel):
    """GET /health — used by Railway + GitHub Actions to confirm liveness."""
    status: str = "ok"
    version: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    services: dict[str, str] = Field(default_factory=dict)
