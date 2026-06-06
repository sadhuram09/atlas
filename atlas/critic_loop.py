"""
atlas/critic_loop.py

CriticLoop — Phase 1 self-healing loop, now integrated with the event store.
"""

from __future__ import annotations

import asyncio

from atlas.agents.coder import CoderAgent
from atlas.agents.verifier import VerifierAgent
from atlas.api.task_service import TaskService
from atlas.api.websocket_manager import ws_manager
from atlas.contracts import (
    AgentMessage,
    AgentRole,
    Artifact,
    EventType,
    StreamEvent,
    TaskRequest,
    TaskStatus,
    TestResult,
)
from atlas.contracts_v2 import EventKind
from atlas.event_store import event_store
from atlas.logging import get_logger

log = get_logger(__name__)


class CriticLoop:
    def __init__(self, task_id: str, task_service: TaskService) -> None:
        self.task_id = task_id
        self.svc = task_service
        self.log = get_logger(__name__).bind(task_id=task_id)

    async def run(self, request: TaskRequest) -> None:
        self.log.info("critic_loop_started", max_retries=request.max_retries)

        # Record task accepted in event store
        event_store.append(
            self.task_id,
            EventKind.TASK_ACCEPTED,
            payload={"title": request.title, "language": request.language, "path": "fast_path"},
        )

        # Initialize governor budget for fast-path tasks
        from atlas.governor.governor import governor
        governor.initialize_task(self.task_id, request.budget_usd)

        previous_code: str = ""
        test_failure: str = ""
        test_result: TestResult | None = None

        for attempt in range(request.max_retries + 1):
            self.log.info("critic_loop_attempt", attempt=attempt)

            # ── CODING ────────────────────────────────────────────────
            await self.svc.update_status(self.task_id, TaskStatus.CODING)
            await self._emit(EventType.AGENT_STARTED, {"agent": AgentRole.CODER, "attempt": attempt})

            event_store.append(
                self.task_id, EventKind.AGENT_CALLED,
                agent=AgentRole.CODER,
                payload={"attempt": attempt},
            )

            coder_msg = AgentMessage(
                task_id=self.task_id,
                from_agent=AgentRole.ORCHESTRATOR,
                to_agent=AgentRole.CODER,
                payload={
                    "task_title": request.title,
                    "task_description": request.description,
                    "language": request.language,
                    "attempt": attempt,
                    "previous_code": previous_code,
                    "test_failure": test_failure,
                },
            )

            try:
                async with CoderAgent(self.task_id) as coder:
                    coder_result = await coder.run(coder_msg)
            except Exception as e:
                self.log.error("coder_agent_crashed", error=str(e), attempt=attempt)
                await self._emit(EventType.AGENT_FAILED, {"agent": AgentRole.CODER, "error": str(e)})
                await self._fail(f"CoderAgent crashed: {e}")
                return

            artifacts_raw = coder_result.payload.get("artifacts", [])
            explanation = coder_result.payload.get("explanation", "")

            event_store.append(
                self.task_id, EventKind.AGENT_RESPONDED,
                agent=AgentRole.CODER,
                payload={"attempt": attempt, "artifacts": [a["filename"] for a in artifacts_raw], "explanation": explanation},
            )

            await self._emit(EventType.AGENT_COMPLETED, {
                "agent": AgentRole.CODER,
                "attempt": attempt,
                "artifacts": [a["filename"] for a in artifacts_raw],
                "explanation": explanation,
            })

            impl = next((a for a in artifacts_raw if not a["filename"].startswith("test_")), None)
            if impl:
                previous_code = impl["content"]

            for a in artifacts_raw:
                await self.svc.add_artifact(self.task_id, Artifact(**a))

            # ── VERIFYING ─────────────────────────────────────────────
            await self.svc.update_status(self.task_id, TaskStatus.VERIFYING)
            await self._emit(EventType.AGENT_STARTED, {"agent": AgentRole.VERIFIER, "attempt": attempt})

            event_store.append(
                self.task_id, EventKind.AGENT_CALLED,
                agent=AgentRole.VERIFIER,
                payload={"attempt": attempt},
            )

            verifier_msg = AgentMessage(
                task_id=self.task_id,
                from_agent=AgentRole.ORCHESTRATOR,
                to_agent=AgentRole.VERIFIER,
                payload={"artifacts": artifacts_raw, "attempt": attempt},
            )

            try:
                async with VerifierAgent(self.task_id) as verifier:
                    verifier_result = await verifier.run(verifier_msg)
            except Exception as e:
                self.log.error("verifier_agent_crashed", error=str(e), attempt=attempt)
                await self._emit(EventType.AGENT_FAILED, {"agent": AgentRole.VERIFIER, "error": str(e)})
                await self._fail(f"VerifierAgent crashed: {e}")
                return

            v_payload = verifier_result.payload
            passed: bool = v_payload["passed"]
            test_result = TestResult(**v_payload["test_result"])
            test_failure = test_result.stdout + "\n" + test_result.stderr

            event_store.append(
                self.task_id, EventKind.SUBTASK_VERIFIED,
                agent=AgentRole.VERIFIER,
                payload={
                    "passed": passed,
                    "attempt": attempt,
                    "test_count": test_result.test_count,
                    "failed_tests": test_result.failed_tests,
                    "duration_ms": test_result.duration_ms,
                },
            )

            await self._emit(EventType.TEST_RESULT, {
                "passed": passed,
                "attempt": attempt,
                "test_count": test_result.test_count,
                "failed_tests": test_result.failed_tests,
                "stdout": test_result.stdout,
                "duration_ms": test_result.duration_ms,
            })
            await self._emit(EventType.AGENT_COMPLETED, {"agent": AgentRole.VERIFIER, "passed": passed, "attempt": attempt})

            # ── VERDICT ───────────────────────────────────────────────
            if passed:
                await self.svc.update_status(self.task_id, TaskStatus.COMPLETED)
                event_store.append(
                    self.task_id, EventKind.TASK_COMPLETED,
                    payload={"attempt": attempt, "test_count": test_result.test_count},
                )
                await self._emit(EventType.TASK_UPDATED, {
                    "status": TaskStatus.COMPLETED,
                    "attempt": attempt,
                    "test_count": test_result.test_count,
                    "message": f"All {test_result.test_count} tests passed on attempt {attempt + 1}",
                })
                self.log.info("critic_loop_success", attempt=attempt, test_count=test_result.test_count)
                return

            retries_left = request.max_retries - attempt - 1
            self.log.warning("critic_loop_retry", attempt=attempt, retries_left=retries_left, failed_tests=test_result.failed_tests)

            if retries_left > 0:
                await self.svc.update_status(self.task_id, TaskStatus.RETRY)
                event_store.append(
                    self.task_id, EventKind.SUBTASK_RETRY,
                    payload={"attempt": attempt, "retries_left": retries_left, "failed_tests": test_result.failed_tests},
                )
                await self._emit(EventType.TASK_UPDATED, {
                    "status": TaskStatus.RETRY,
                    "attempt": attempt,
                    "retries_left": retries_left,
                    "failed_tests": test_result.failed_tests,
                })
                await asyncio.sleep(2)

        failed_tests = test_result.failed_tests if test_result else []
        await self._fail(f"Exhausted {request.max_retries} retries. Last failure: {failed_tests}")

    async def _emit(self, event_type: EventType, data: dict) -> None:
        await ws_manager.broadcast(StreamEvent(event=event_type, task_id=self.task_id, data=data))

    async def _fail(self, reason: str) -> None:
        await self.svc.update_status(self.task_id, TaskStatus.FAILED, error=reason)
        event_store.append(self.task_id, EventKind.TASK_FAILED, payload={"reason": reason})
        await self._emit(EventType.TASK_UPDATED, {"status": TaskStatus.FAILED, "error": reason})
        self.log.error("critic_loop_failed", reason=reason)
