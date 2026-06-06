"""
atlas/event_store.py

Immutable append-only event store.

Every state change in ATLAS produces an ExecutionEvent.
Events are never deleted or modified — only appended.

Phase 2: In-memory (list per task_id).
Phase 3+: Swap for PostgreSQL — the interface stays identical.

Why event sourcing?
  Traditional approach: store current state (status = "completed").
  Event sourcing: store what happened (task.accepted → subtask.started → ...)

  Benefits:
    1. Complete audit trail — you know exactly how a task got to its state
    2. Time travel — replay events to reconstruct state at any point in time
    3. The DAG visualiser subscribes to this stream directly
    4. Debugging is trivial — play back the event sequence

  For ATLAS specifically: when an agent makes a decision, we want to know
  WHY it made that decision. Events capture the inputs and outputs
  of every agent call, not just the final state.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from atlas.contracts_v2 import EventKind, ExecutionEvent
from atlas.contracts import AgentRole
from atlas.logging import get_logger

log = get_logger(__name__)


class EventStore:
    """
    Append-only in-memory event store.

    One singleton per server process — all tasks share the same store.
    Thread-safe for asyncio (single event loop, GIL-protected dict ops).
    """

    def __init__(self) -> None:
        # task_id → ordered list of events
        self._events: dict[str, list[ExecutionEvent]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        task_id: str,
        kind: EventKind,
        payload: dict[str, Any] | None = None,
        *,
        agent: AgentRole | None = None,
        subtask_id: str | None = None,
        wave: int | None = None,
        duration_ms: int | None = None,
    ) -> ExecutionEvent:
        """
        Append an event to the store and return it.

        This is the ONLY way to write events. No updates, no deletes.
        """
        event = ExecutionEvent(
            task_id=task_id,
            kind=kind,
            agent=agent,
            subtask_id=subtask_id,
            wave=wave,
            payload=payload or {},
            duration_ms=duration_ms,
        )
        self._events[task_id].append(event)

        log.debug(
            "event_appended",
            task_id=task_id,
            kind=kind,
            event_id=event.id,
        )

        return event

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self, task_id: str) -> list[ExecutionEvent]:
        """Return all events for a task, in order."""
        return list(self._events.get(task_id, []))

    def get_by_kind(self, task_id: str, kind: EventKind) -> list[ExecutionEvent]:
        """Return all events of a specific kind for a task."""
        return [e for e in self._events.get(task_id, []) if e.kind == kind]

    def get_by_subtask(self, task_id: str, subtask_id: str) -> list[ExecutionEvent]:
        """Return all events for a specific subtask."""
        return [
            e for e in self._events.get(task_id, [])
            if e.subtask_id == subtask_id
        ]

    def last_event(self, task_id: str) -> ExecutionEvent | None:
        """Return the most recent event for a task."""
        events = self._events.get(task_id, [])
        return events[-1] if events else None

    def count(self, task_id: str) -> int:
        return len(self._events.get(task_id, []))

    # ------------------------------------------------------------------
    # Replay / reconstruction
    # ------------------------------------------------------------------

    def timeline(self, task_id: str) -> list[dict]:
        """
        Return a human-readable timeline of events.

        Used by GET /tasks/{id}/timeline endpoint.
        The frontend DAG visualiser renders this.
        """
        events = self.get_all(task_id)
        return [
            {
                "id": e.id,
                "kind": e.kind,
                "agent": e.agent,
                "subtask_id": e.subtask_id,
                "wave": e.wave,
                "timestamp": e.timestamp.isoformat(),
                "duration_ms": e.duration_ms,
                "payload": e.payload,
            }
            for e in events
        ]

    def stats(self, task_id: str) -> dict:
        """Aggregate stats for a task — used by the dashboard."""
        events = self.get_all(task_id)
        if not events:
            return {}

        start = events[0].timestamp
        end = events[-1].timestamp
        duration = int((end - start).total_seconds() * 1000)

        # Count tokens from agent.responded events (where token data lives)
        agent_responses = [e for e in events if e.kind == EventKind.AGENT_RESPONDED]
        llm_calls = [e for e in events if e.kind == EventKind.LLM_CALL_MADE]
        total_tokens = sum(
            e.payload.get("tokens_in", 0) + e.payload.get("tokens_out", 0)
            for e in llm_calls + agent_responses
        )
        llm_call_count = len(llm_calls) + len(agent_responses)
        waves = {e.wave for e in events if e.wave is not None}

        return {
            "total_events": len(events),
            "llm_calls": llm_call_count,
            "total_tokens": total_tokens,
            "waves_executed": len(waves),
            "duration_ms": duration,
            "started_at": start.isoformat(),
            "ended_at": end.isoformat(),
        }


# Singleton — shared across the entire application
event_store = EventStore()
