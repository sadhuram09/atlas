"""
tests/unit/test_contracts.py

Unit tests for Pydantic contracts — verify validation rules are correct.

These tests import contracts directly (no HTTP layer).
Fast, pure, no I/O.
"""

import pytest
from pydantic import ValidationError

from atlas.contracts import (
    AgentMessage,
    AgentRole,
    Artifact,
    EventType,
    ModelTier,
    StreamEvent,
    SubTask,
    TaskRequest,
    TaskStatus,
)


class TestTaskRequest:
    def test_valid_request(self) -> None:
        req = TaskRequest(
            title="Write a sort function",
            description="Write a merge sort in Python with type hints",
            language="python",
        )
        assert req.title == "Write a sort function"
        assert req.max_retries == 3  # Default
        assert req.budget_usd == 0.50  # Default

    def test_title_required(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(title="", description="x" * 20)

    def test_description_min_length(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(title="Test", description="short")

    def test_budget_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(
                title="Test",
                description="A valid description that is long enough",
                budget_usd=0.0,
            )

    def test_budget_max_limit(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(
                title="Test",
                description="A valid description that is long enough",
                budget_usd=100.0,  # Over the 10.0 limit
            )

    def test_max_retries_range(self) -> None:
        with pytest.raises(ValidationError):
            TaskRequest(
                title="Test",
                description="A valid description that is long enough",
                max_retries=0,  # Must be >= 1
            )


class TestEnums:
    def test_task_status_values(self) -> None:
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.COMPLETED == "completed"

    def test_agent_role_values(self) -> None:
        assert AgentRole.CODER == "coder"
        assert AgentRole.VERIFIER == "verifier"

    def test_model_tier_values(self) -> None:
        assert ModelTier.FAST == "fast"
        assert ModelTier.BALANCED == "balanced"
        assert ModelTier.POWERFUL == "powerful"


class TestAgentMessage:
    def test_message_gets_auto_id(self) -> None:
        msg = AgentMessage(
            task_id="test-task",
            from_agent=AgentRole.ORCHESTRATOR,
            to_agent=AgentRole.CODER,
            payload={"instruction": "Write a function"},
        )
        assert len(msg.id) == 36  # UUID

    def test_message_is_frozen(self) -> None:
        """Frozen models can't be mutated — ensures auditability."""
        msg = AgentMessage(
            task_id="test-task",
            from_agent=AgentRole.ORCHESTRATOR,
            to_agent=AgentRole.CODER,
            payload={},
        )
        with pytest.raises(Exception):  # ValidationError or TypeError
            msg.task_id = "mutated"  # type: ignore


class TestStreamEvent:
    def test_stream_event_creation(self) -> None:
        event = StreamEvent(
            event=EventType.TASK_CREATED,
            task_id="test-123",
            data={"title": "Test task"},
        )
        assert event.event == "task.created"
        assert event.task_id == "test-123"

    def test_stream_event_serialises_to_json(self) -> None:
        event = StreamEvent(
            event=EventType.AGENT_STARTED,
            task_id="test-123",
        )
        data = event.model_dump(mode="json")
        assert isinstance(data["timestamp"], str)
        assert data["event"] == "agent.started"
