"""
atlas/agents/verifier.py

VerifierAgent — the L1 gate. Nothing passes upward until this says yes.

Core invariant (from ATLAS spec):
  "Nothing propagates up until it passes the L1 verification gate."

What it does:
  1. Receives artifacts from CoderAgent
  2. Passes them to DockerSandbox
  3. Returns a verdict: PASS or FAIL
  4. On FAIL: includes the full pytest output so CoderAgent can fix it

Why a separate agent?
  - Single responsibility: verification is separate from generation
  - Swappable: you can plug in different sandboxes (Docker, E2B, Modal)
  - Observable: every verification emits a structured TestResult
  - The critic loop in CriticLoop depends only on this agent's output format

The verifier itself never calls an LLM. It's pure deterministic execution.
LLMs generate. Docker verifies. Clear boundary.
"""

from __future__ import annotations

from atlas.agents.base import BaseAgent
from atlas.contracts import (
    AgentMessage,
    AgentRole,
    Artifact,
    TestResult,
)
from atlas.tools.sandbox import DockerSandbox
from atlas.logging import get_logger

log = get_logger(__name__)


class VerifierAgent(BaseAgent):
    """
    Runs generated code in Docker and returns a pass/fail verdict.

    No LLM calls — pure deterministic execution.
    The fastest agent in the system.
    """

    role = AgentRole.VERIFIER

    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        self.sandbox = DockerSandbox()

    async def run(self, message: AgentMessage) -> AgentMessage:
        """
        Run artifacts through the sandbox and return verdict.

        Payload in:
            artifacts: list[dict]   (serialised Artifact objects from CoderAgent)
            attempt: int

        Payload out:
            passed: bool
            test_result: dict       (serialised TestResult)
            artifacts: list[dict]   (passed through unchanged)
            attempt: int
        """
        payload = message.payload
        attempt = payload.get("attempt", 0)

        # Deserialise artifacts from dict → Artifact objects
        raw_artifacts = payload.get("artifacts", [])
        artifacts = [Artifact(**a) for a in raw_artifacts]

        self.log.info(
            "verifier_started",
            attempt=attempt,
            files=[a.filename for a in artifacts],
        )

        # Run in sandbox — this is the core gate
        test_result: TestResult = self.sandbox.run(artifacts)

        if test_result.passed:
            self.log.info(
                "verifier_passed",
                attempt=attempt,
                test_count=test_result.test_count,
                duration_ms=test_result.duration_ms,
            )
        else:
            self.log.warning(
                "verifier_failed",
                attempt=attempt,
                exit_code=test_result.exit_code,
                failed_tests=test_result.failed_tests,
                duration_ms=test_result.duration_ms,
            )

        return self._reply(
            message,
            payload={
                "passed": test_result.passed,
                "test_result": test_result.model_dump(),
                "artifacts": raw_artifacts,  # Pass through unchanged
                "attempt": attempt,
            },
        )
