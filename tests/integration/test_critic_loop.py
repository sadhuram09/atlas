"""
tests/integration/test_critic_loop.py

Integration test — runs the REAL critic loop with a real Groq call.

This test:
  1. Creates a simple coding task
  2. Runs CoderAgent (calls Groq — uses your GROQ_API_KEY)
  3. Runs VerifierAgent (runs pytest directly — no Docker needed)
  4. Asserts the task reaches COMPLETED status

Skip condition:
  If GROQ_API_KEY is the placeholder value, this test is skipped.
  Run it explicitly: pytest tests/integration/ -v

Time: ~10-30 seconds (Groq is fast)
Cost: $0.00 (Groq free tier)
"""

import os
import asyncio
import pytest

from atlas.api.task_service import TaskService
from atlas.contracts import TaskRequest, TaskStatus
from atlas.critic_loop import CriticLoop


def has_real_groq_key() -> bool:
    key = os.getenv("GROQ_API_KEY", "gsk_placeholder")
    return key != "gsk_placeholder" and key.startswith("gsk_")


@pytest.mark.skipif(
    not has_real_groq_key(),
    reason="GROQ_API_KEY not set — skipping integration test",
)
@pytest.mark.asyncio
async def test_full_critic_loop_simple_task() -> None:
    """
    Full end-to-end: submit a simple task, verify it reaches COMPLETED.

    Uses a trivially simple task (add two numbers) to minimise Groq usage
    and maximise the chance of passing on first attempt.
    """
    svc = TaskService()

    request = TaskRequest(
        title="Add two numbers",
        description=(
            "Write a Python function called `add(a: int, b: int) -> int` "
            "that returns the sum of two integers. "
            "Include a docstring and handle both positive and negative numbers."
        ),
        language="python",
        max_retries=3,
        budget_usd=1.0,
    )

    # Create the task
    response = await svc.create(request)
    task_id = response.task_id

    # Run the critic loop
    loop = CriticLoop(task_id=task_id, task_service=svc)
    await loop.run(request)

    # Check final state
    task = await svc.get(task_id)

    assert task.status == TaskStatus.COMPLETED, (
        f"Expected COMPLETED but got {task.status}. "
        f"Error: {task.error}"
    )
    assert len(task.artifacts) >= 2, "Should have solution.py and test_solution.py"

    filenames = [a.filename for a in task.artifacts]
    assert "solution.py" in filenames
    assert "test_solution.py" in filenames


@pytest.mark.skipif(
    not has_real_groq_key(),
    reason="GROQ_API_KEY not set — skipping integration test",
)
@pytest.mark.asyncio
async def test_critic_loop_produces_artifacts() -> None:
    """Verify that artifacts have content and checksums."""
    svc = TaskService()

    request = TaskRequest(
        title="Reverse a string",
        description=(
            "Write a Python function called `reverse_string(s: str) -> str` "
            "that returns the input string reversed. "
            "Handle empty strings correctly."
        ),
        language="python",
        max_retries=3,
        budget_usd=1.0,
    )

    response = await svc.create(request)
    loop = CriticLoop(task_id=response.task_id, task_service=svc)
    await loop.run(request)

    task = await svc.get(response.task_id)

    for artifact in task.artifacts:
        assert artifact.content, f"{artifact.filename} has empty content"
        assert artifact.checksum, f"{artifact.filename} has no checksum"
        assert len(artifact.checksum) == 64, "Checksum should be SHA-256 hex"
