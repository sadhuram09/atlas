"""
atlas/orchestrator.py

Orchestrator — Phase 3 updated with memory + governor integration.

New in Phase 3:
  - Governor approves every model tier before calls
  - FailureMemory stores fix patterns after successful retries
  - PromptEnhancer enriches retry prompts with past failure context
  - Budget state broadcast to WebSocket on every LLM call
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

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
from atlas.contracts_v2 import (
    EventKind,
    ExecutionPlan,
    PipelineResult,
    SubTaskNode,
    SubTaskStatus,
)
from atlas.contracts_v3 import FailurePattern
from atlas.event_store import event_store
from atlas.governor.governor import governor
from atlas.memory.failure_memory import failure_memory
from atlas.memory.prompt_enhancer import prompt_enhancer
from atlas.logging import get_logger

log = get_logger(__name__)

SUBTASK_MAX_RETRIES = 3


class Orchestrator:
    """
    Executes a DAG of subtasks with parallel wave execution.
    Phase 3: Memory + Governor integrated into every subtask execution.
    """

    def __init__(self, task_id: str, task_service: TaskService) -> None:
        self.task_id = task_id
        self.svc = task_service
        self.log = get_logger(__name__).bind(task_id=task_id)

    async def execute(self, plan: ExecutionPlan, request: TaskRequest) -> PipelineResult:
        start_time = time.monotonic()
        total_attempts = 0
        all_artifacts: list[Artifact] = []

        # Initialize Governor budget for this task
        governor.initialize_task(self.task_id, request.budget_usd)

        self.log.info(
            "orchestrator_started",
            waves=len(plan.execution_waves),
            subtasks=len(plan.nodes),
        )

        node_status: dict[str, SubTaskStatus] = {n.id: n.status for n in plan.nodes}

        for wave_idx, wave_node_ids in enumerate(plan.execution_waves):
            self.log.info("wave_started", wave=wave_idx, nodes=wave_node_ids)

            event_store.append(
                self.task_id, EventKind.WAVE_STARTED, wave=wave_idx,
                payload={"node_ids": wave_node_ids, "parallel": len(wave_node_ids) > 1},
            )
            await self._ws(EventType.STEP_CREATED, {
                "wave": wave_idx,
                "node_ids": wave_node_ids,
                "parallel": len(wave_node_ids) > 1,
            })

            wave_nodes = [n for n in plan.nodes if n.id in wave_node_ids]

            for node in wave_nodes:
                node_status[node.id] = SubTaskStatus.RUNNING
                event_store.append(
                    self.task_id, EventKind.SUBTASK_STARTED,
                    subtask_id=node.id, wave=wave_idx,
                    payload={"title": node.title, "tier": node.assigned_tier},
                )
                await self._ws(EventType.AGENT_STARTED, {
                    "subtask_id": node.id,
                    "subtask_title": node.title,
                    "wave": wave_idx,
                    "tier": node.assigned_tier,
                })

            # Parallel execution
            results = await asyncio.gather(
                *[self._execute_subtask(node, request, wave_idx) for node in wave_nodes],
                return_exceptions=True,
            )

            wave_failed = False

            for node, result in zip(wave_nodes, results):
                if isinstance(result, Exception):
                    node_status[node.id] = SubTaskStatus.FAILED
                    wave_failed = True
                    event_store.append(
                        self.task_id, EventKind.SUBTASK_FAILED,
                        subtask_id=node.id, wave=wave_idx,
                        payload={"error": str(result)},
                    )
                    await self._ws(EventType.AGENT_FAILED, {
                        "subtask_id": node.id, "error": str(result), "wave": wave_idx,
                    })

                elif result["passed"]:
                    node_status[node.id] = SubTaskStatus.PASSED
                    total_attempts += result["attempts"]
                    artifacts = result["artifacts"]
                    all_artifacts.extend(artifacts)

                    for artifact in artifacts:
                        await self.svc.add_artifact(self.task_id, artifact)

                    # Emit budget update
                    budget = governor.get_budget(self.task_id)
                    if budget:
                        await self._ws(EventType.COST_UPDATE, {
                            "task_id": self.task_id,
                            "spent_usd": round(budget.spent_usd, 6),
                            "budget_usd": budget.budget_usd,
                            "percent_used": round(budget.percent_used, 1),
                            "llm_calls": budget.llm_calls,
                            "tokens_in": budget.tokens_in,
                            "tokens_out": budget.tokens_out,
                        })

                    event_store.append(
                        self.task_id, EventKind.SUBTASK_PASSED,
                        subtask_id=node.id, wave=wave_idx,
                        payload={
                            "attempts": result["attempts"],
                            "test_count": result["test_count"],
                            "artifacts": [a.filename for a in artifacts],
                            "memory_patterns_used": result.get("memory_patterns_used", 0),
                        },
                    )
                    await self._ws(EventType.STEP_RESULT, {
                        "subtask_id": node.id,
                        "subtask_title": node.title,
                        "passed": True,
                        "attempts": result["attempts"],
                        "test_count": result["test_count"],
                        "wave": wave_idx,
                        "memory_patterns_used": result.get("memory_patterns_used", 0),
                    })

                else:
                    node_status[node.id] = SubTaskStatus.FAILED
                    wave_failed = True
                    total_attempts += result["attempts"]
                    event_store.append(
                        self.task_id, EventKind.SUBTASK_FAILED,
                        subtask_id=node.id, wave=wave_idx,
                        payload={"attempts": result["attempts"], "failed_tests": result.get("failed_tests", [])},
                    )
                    await self._ws(EventType.AGENT_FAILED, {
                        "subtask_id": node.id,
                        "subtask_title": node.title,
                        "passed": False,
                        "wave": wave_idx,
                        "failed_tests": result.get("failed_tests", []),
                    })

            event_store.append(
                self.task_id, EventKind.WAVE_COMPLETED, wave=wave_idx,
                payload={
                    "passed": not wave_failed,
                    "node_statuses": {nid: node_status[nid] for nid in wave_node_ids},
                },
            )

            if wave_failed:
                break

            self.log.info("wave_completed", wave=wave_idx)

        passed_count = sum(1 for s in node_status.values() if s == SubTaskStatus.PASSED)
        failed_count = sum(1 for s in node_status.values() if s == SubTaskStatus.FAILED)
        success = failed_count == 0 and passed_count == len(plan.nodes)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        return PipelineResult(
            task_id=self.task_id,
            success=success,
            total_subtasks=len(plan.nodes),
            passed_subtasks=passed_count,
            failed_subtasks=failed_count,
            total_attempts=total_attempts,
            all_artifacts=all_artifacts,
            waves_executed=len(plan.execution_waves),
            total_duration_ms=duration_ms,
            complexity_score=plan.complexity.score,
        )

    async def _execute_subtask(
        self, node: SubTaskNode, request: TaskRequest, wave: int,
    ) -> dict[str, Any]:
        """
        Run a single subtask through Coder → Verifier critic loop.
        Phase 3: memory-enhanced retry prompts + governor routing.
        """
        previous_code = ""
        test_failure = ""
        memory_patterns_used = 0
        last_test_result: TestResult | None = None

        for attempt in range(SUBTASK_MAX_RETRIES):
            # ── GOVERNOR: approve model tier ──────────────────────────
            decision = governor.route(
                task_id=self.task_id,
                requested_tier=node.assigned_tier,
                context=f"subtask {node.id} attempt {attempt}",
            )

            # ── MEMORY: enhance retry prompt ──────────────────────────
            enhanced_description = node.description
            if attempt > 0 and test_failure:
                enhanced = prompt_enhancer.enhance_retry_prompt(
                    original_prompt=node.description,
                    error_output=test_failure,
                    language=request.language,
                )
                if enhanced.enhancement_applied:
                    enhanced_description = enhanced.enhanced_prompt
                    memory_patterns_used = enhanced.patterns_used
                    event_store.append(
                        self.task_id, EventKind.AGENT_CALLED,
                        agent=AgentRole.CODER, subtask_id=node.id, wave=wave,
                        payload={
                            "attempt": attempt,
                            "memory_enhanced": True,
                            "patterns_used": enhanced.patterns_used,
                            "top_similarity": enhanced.memory_context[:50] if enhanced.memory_context else "",
                        },
                    )

            event_store.append(
                self.task_id, EventKind.AGENT_CALLED,
                agent=AgentRole.CODER, subtask_id=node.id, wave=wave,
                payload={
                    "attempt": attempt,
                    "tier": decision.approved_tier,
                    "downgraded": decision.downgraded,
                },
            )

            coder_msg = AgentMessage(
                task_id=self.task_id,
                from_agent=AgentRole.ORCHESTRATOR,
                to_agent=AgentRole.CODER,
                payload={
                    "task_title": node.title,
                    "task_description": enhanced_description,
                    "language": request.language,
                    "attempt": attempt,
                    "previous_code": previous_code,
                    "test_failure": test_failure,
                },
            )

            try:
                async with CoderAgent(self.task_id) as coder:
                    coder_result = await coder.run(coder_msg)

                    # Record usage in Governor
                    for cost_event in coder._cost_events:
                        governor.record_usage(
                            self.task_id,
                            cost_event.tokens_in,
                            cost_event.tokens_out,
                            cost_event.model,
                        )
            except Exception as e:
                self.log.error("subtask_coder_crashed", subtask_id=node.id, attempt=attempt, error=str(e))
                continue

            artifacts_raw = coder_result.payload.get("artifacts", [])
            impl = next((a for a in artifacts_raw if not a["filename"].startswith("test_")), None)
            if impl:
                previous_code = impl["content"]

            event_store.append(
                self.task_id, EventKind.AGENT_RESPONDED,
                agent=AgentRole.CODER, subtask_id=node.id, wave=wave,
                payload={"attempt": attempt, "artifacts": [a["filename"] for a in artifacts_raw]},
            )

            # ── VERIFIER ──────────────────────────────────────────────
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
                self.log.error("subtask_verifier_crashed", subtask_id=node.id, attempt=attempt, error=str(e))
                continue

            v_payload = verifier_result.payload
            passed = v_payload["passed"]
            test_result = TestResult(**v_payload["test_result"])
            last_test_result = test_result
            test_failure = test_result.stdout + "\n" + test_result.stderr

            event_store.append(
                self.task_id, EventKind.SUBTASK_VERIFIED,
                subtask_id=node.id, wave=wave,
                payload={
                    "passed": passed, "attempt": attempt,
                    "test_count": test_result.test_count,
                    "failed_tests": test_result.failed_tests,
                },
            )

            if passed:
                # ── MEMORY: store fix pattern if this was a retry ─────
                if attempt > 0 and previous_code:
                    failure_memory.store(
                        task_id=self.task_id,
                        subtask_title=node.title,
                        language=request.language,
                        error_output=test_failure,
                        failed_code=previous_code,
                        fixed_code=impl["content"] if impl else previous_code,
                        test_count=test_result.test_count,
                    )

                artifacts = [Artifact(**a) for a in artifacts_raw]
                return {
                    "passed": True,
                    "attempts": attempt + 1,
                    "artifacts": artifacts,
                    "test_count": test_result.test_count,
                    "failed_tests": [],
                    "memory_patterns_used": memory_patterns_used,
                }

            event_store.append(
                self.task_id, EventKind.SUBTASK_RETRY,
                subtask_id=node.id, wave=wave,
                payload={
                    "attempt": attempt,
                    "failed_tests": test_result.failed_tests,
                    "retries_left": SUBTASK_MAX_RETRIES - attempt - 1,
                    "memory_available": failure_memory.pattern_count,
                },
            )
            await asyncio.sleep(1)

        return {
            "passed": False,
            "attempts": SUBTASK_MAX_RETRIES,
            "artifacts": [],
            "test_count": 0,
            "failed_tests": last_test_result.failed_tests if last_test_result else [],
        }

    async def _ws(self, event_type: EventType, data: dict) -> None:
        await ws_manager.broadcast(StreamEvent(event=event_type, task_id=self.task_id, data=data))
