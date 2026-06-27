"""
tests/integration/test_mocked_pipeline.py

Full-pipeline integration tests with mocked LLM and sandbox.
Zero real Groq calls, zero Docker, zero network — fast and deterministic.

Three correctness scenarios:
  happy    — task completes in one shot (baseline)
  b1_exit5 — all attempts fail with pytest exit_code=5 → verify B1 human-readable error
  b2_retry — first sandbox run fails, second passes → verify B2 no duplicate filenames

Load test (scenario 4):
  25 concurrent POST /tasks, cancellation mid-flight, 5 more tasks added mid-load,
  semaphore cap verified, all remaining tasks reach terminal state.

Run:
    poetry run pytest tests/integration/test_mocked_pipeline.py -v -s
"""

from __future__ import annotations

import asyncio
import time
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from atlas.agents.coder import CoderAgent
from atlas.agents.intelligence_router import IntelligenceRouter
from atlas.api.app import create_app
from atlas.contracts import TestResult
from atlas.tools.sandbox import DockerSandbox


# ---------------------------------------------------------------------------
# Mock responses
# ---------------------------------------------------------------------------

# IntelligenceRouter returns this — score ≤ 3 + 1 subtask → fast path (CriticLoop)
_COMPLEXITY_SIMPLE = (
    '{"score": 2, "tier": "fast", "reasoning": "mock", '
    '"estimated_subtasks": 1, "requires_parallel": false}'
)

# CoderAgent returns these — parsed into solution.py + test_solution.py
_CODER_V1 = """\
## Implementation

```python
def add(a: int, b: int) -> int:
    return a + b  # attempt-1
```

## Tests

```python
from solution import add

def test_add():
    assert add(1, 2) == 3
```

## Explanation
Version 1.
"""

_CODER_V2 = """\
## Implementation

```python
def add(a: int, b: int) -> int:
    return a + b  # attempt-2
```

## Tests

```python
from solution import add

def test_add():
    assert add(2, 2) == 4
```

## Explanation
Version 2 — fixed after retry.
"""

# DockerSandbox._run_sync returns these
SANDBOX_PASS = TestResult(
    passed=True, exit_code=0,
    stdout="test_solution.py::test_add PASSED\n1 passed in 0.01s",
    stderr="", test_count=1, failed_tests=[], duration_ms=20,
)

SANDBOX_FAIL_1 = TestResult(
    passed=False, exit_code=1,
    stdout="test_solution.py::test_add FAILED\nAssertionError\n1 failed in 0.01s",
    stderr="", test_count=1, failed_tests=["test_add"], duration_ms=20,
)

SANDBOX_EXIT5 = TestResult(
    passed=False, exit_code=5,
    stdout="collected 0 items\nno tests ran",
    stderr="", test_count=0, failed_tests=[], duration_ms=5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_BODY = {
    "title": "Mock pipeline task",
    "description": "Write an add function with full test coverage and type hints",
    "language": "python",
    "max_retries": 3,
    "budget_usd": 0.50,
}


async def wait_terminal(
    client: httpx.AsyncClient,
    task_id: str,
    timeout: float = 15.0,
) -> dict:
    """Poll GET /tasks/{id} until status is 'completed' or 'failed'."""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        r = await client.get(f"/tasks/{task_id}")
        last = r.json()
        if last["status"] in ("completed", "failed"):
            return last
        await asyncio.sleep(0.05)
    raise TimeoutError(
        f"Task {task_id} still {last.get('status')!r} after {timeout}s"
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _configure_once() -> None:
    """Set up process-level singletons exactly once for the whole test session."""
    from atlas.logging import configure_logging
    configure_logging()


@pytest.fixture
def fresh_app():
    """
    Fresh FastAPI app per test.

    ASGITransport skips the ASGI lifespan, so we pre-initialize app.state.
    TaskService is a plain in-memory dict — safe to construct directly.

    failure_memory is NOT initialized here intentionally: the fast-path tests
    (score ≤ 3 → CriticLoop) never touch it, and loading sentence-transformers
    takes ~25s + crashes on repeated calls within the same process.
    """
    from atlas.api.task_service import TaskService

    app = create_app()
    app.state.task_service = TaskService()
    return app


# ---------------------------------------------------------------------------
# Scenario 1: Happy path — completes in one attempt
# ---------------------------------------------------------------------------

class TestHappyPath:

    async def test_task_reaches_completed(self, fresh_app, monkeypatch) -> None:
        monkeypatch.setattr(IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE))
        monkeypatch.setattr(CoderAgent, "complete", AsyncMock(return_value=_CODER_V1))
        monkeypatch.setattr(DockerSandbox, "_run_sync", MagicMock(return_value=SANDBOX_PASS))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh_app), base_url="http://test") as client:
            r = await client.post("/tasks", json=TASK_BODY)
            assert r.status_code == 202
            task_id = r.json()["task_id"]
            task = await wait_terminal(client, task_id)

        assert task["status"] == "completed", f"Expected completed, got: {task}"

    async def test_artifacts_present_on_completion(self, fresh_app, monkeypatch) -> None:
        monkeypatch.setattr(IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE))
        monkeypatch.setattr(CoderAgent, "complete", AsyncMock(return_value=_CODER_V1))
        monkeypatch.setattr(DockerSandbox, "_run_sync", MagicMock(return_value=SANDBOX_PASS))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh_app), base_url="http://test") as client:
            r = await client.post("/tasks", json=TASK_BODY)
            task = await wait_terminal(client, r.json()["task_id"])

        filenames = {a["filename"] for a in task["artifacts"]}
        assert "solution.py" in filenames
        assert "test_solution.py" in filenames

    async def test_coder_called_exactly_once_on_first_pass(self, fresh_app, monkeypatch) -> None:
        coder_mock = AsyncMock(return_value=_CODER_V1)
        sandbox_mock = MagicMock(return_value=SANDBOX_PASS)

        monkeypatch.setattr(IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE))
        monkeypatch.setattr(CoderAgent, "complete", coder_mock)
        monkeypatch.setattr(DockerSandbox, "_run_sync", sandbox_mock)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh_app), base_url="http://test") as client:
            r = await client.post("/tasks", json=TASK_BODY)
            await wait_terminal(client, r.json()["task_id"])

        assert coder_mock.call_count == 1
        assert sandbox_mock.call_count == 1


# ---------------------------------------------------------------------------
# Scenario 2: B1 — pytest exit code 5 produces human-readable error
# ---------------------------------------------------------------------------

class TestB1ExitCode5Message:
    """
    Mock the sandbox to always return exit_code=5 (no tests collected).
    After exhausting max_retries, the task's error field must be the
    human-readable message from critic_loop.py, NOT "Last failure: []".
    """

    async def test_b1_error_is_human_readable(self, fresh_app, monkeypatch) -> None:
        # All sandbox calls return exit_code=5 — no tests collected
        monkeypatch.setattr(IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE))
        monkeypatch.setattr(CoderAgent, "complete", AsyncMock(return_value=_CODER_V1))
        monkeypatch.setattr(DockerSandbox, "_run_sync", MagicMock(return_value=SANDBOX_EXIT5))

        # max_retries=1 → 2 total attempts (attempt 0 + attempt 1), then exhausted
        body = {**TASK_BODY, "max_retries": 1}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh_app), base_url="http://test") as client:
            r = await client.post("/tasks", json=body)
            task_id = r.json()["task_id"]
            task = await wait_terminal(client, task_id)

        print(f"\n[B1] Final error message:\n  {task.get('error')}")

        assert task["status"] == "failed"
        error = task.get("error", "")

        # B1 FIX: must NOT show old "Last failure: []"
        assert "[]" not in error, f"B1 NOT FIXED — old format leak: {error!r}"

        # B1 FIX: must show the human-readable exit_code=5 message
        expected_fragment = "no tests collected"
        assert expected_fragment in error, (
            f"B1 NOT FIXED — expected {expected_fragment!r} in error, got: {error!r}"
        )

        # Full expected string
        assert "LLM did not generate any test functions" in error, (
            f"B1 message not specific enough: {error!r}"
        )

    async def test_b1_error_contains_retry_count(self, fresh_app, monkeypatch) -> None:
        monkeypatch.setattr(IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE))
        monkeypatch.setattr(CoderAgent, "complete", AsyncMock(return_value=_CODER_V1))
        monkeypatch.setattr(DockerSandbox, "_run_sync", MagicMock(return_value=SANDBOX_EXIT5))

        body = {**TASK_BODY, "max_retries": 1}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh_app), base_url="http://test") as client:
            r = await client.post("/tasks", json=body)
            task = await wait_terminal(client, r.json()["task_id"])

        # Should mention exhausted retries count
        assert "Exhausted 1 retries" in task["error"], (
            f"Expected 'Exhausted 1 retries' in: {task['error']!r}"
        )


# ---------------------------------------------------------------------------
# Scenario 3: B2 — no duplicate filenames on retry
# ---------------------------------------------------------------------------

class TestB2NoDuplicateArtifacts:
    """
    First sandbox run fails, second passes.
    CriticLoop retries and calls CoderAgent again.
    Both attempts produce solution.py + test_solution.py.

    B2 fix (add_artifact dedup): final artifacts must have exactly 2 entries,
    both with unique filenames — the LATEST version for each file.
    """

    async def test_b2_no_duplicate_filenames_after_retry(self, fresh_app, monkeypatch) -> None:
        # First coder call → v1, second call → v2 (different content, same filenames)
        coder_mock = AsyncMock(side_effect=[_CODER_V1, _CODER_V2])
        # First sandbox call → FAIL, second → PASS
        sandbox_mock = MagicMock(side_effect=[SANDBOX_FAIL_1, SANDBOX_PASS])

        monkeypatch.setattr(IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE))
        monkeypatch.setattr(CoderAgent, "complete", coder_mock)
        monkeypatch.setattr(DockerSandbox, "_run_sync", sandbox_mock)

        body = {**TASK_BODY, "max_retries": 2}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh_app), base_url="http://test") as client:
            r = await client.post("/tasks", json=body)
            task_id = r.json()["task_id"]
            task = await wait_terminal(client, task_id)

        artifacts = task["artifacts"]
        filenames = [a["filename"] for a in artifacts]

        print(f"\n[B2] Artifacts after retry:")
        for a in artifacts:
            preview = a["content"].split("\n")[1] if "\n" in a["content"] else a["content"][:40]
            print(f"  {a['filename']}: {preview!r}")
        print(f"  Retry count: coder called {coder_mock.call_count}x, sandbox {sandbox_mock.call_count}x")

        assert task["status"] == "completed", f"Expected completed, got {task['status']}: {task.get('error')}"

        # B2: exactly 2 artifacts, no duplicates
        assert len(filenames) == 2, (
            f"B2 NOT FIXED — {len(filenames)} artifacts found (duplicates?): {filenames}"
        )
        assert len(set(filenames)) == 2, (
            f"B2 NOT FIXED — duplicate filenames: {filenames}"
        )

        # Verify it's the LATEST version (v2) that survived
        sol = next(a for a in artifacts if a["filename"] == "solution.py")
        assert "attempt-2" in sol["content"], (
            f"B2: expected v2 content to survive retry, got: {sol['content'][:60]!r}"
        )

    async def test_b2_coder_called_twice_on_single_retry(self, fresh_app, monkeypatch) -> None:
        coder_mock = AsyncMock(side_effect=[_CODER_V1, _CODER_V2])
        sandbox_mock = MagicMock(side_effect=[SANDBOX_FAIL_1, SANDBOX_PASS])

        monkeypatch.setattr(IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE))
        monkeypatch.setattr(CoderAgent, "complete", coder_mock)
        monkeypatch.setattr(DockerSandbox, "_run_sync", sandbox_mock)

        body = {**TASK_BODY, "max_retries": 2}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fresh_app), base_url="http://test") as client:
            r = await client.post("/tasks", json=body)
            await wait_terminal(client, r.json()["task_id"])

        assert coder_mock.call_count == 2, f"Expected 2 coder calls, got {coder_mock.call_count}"
        assert sandbox_mock.call_count == 2, f"Expected 2 sandbox calls, got {sandbox_mock.call_count}"


# ---------------------------------------------------------------------------
# Scenario 4: Load test — 25 concurrent tasks + mid-flight cancellation + mid-load additions
# ---------------------------------------------------------------------------

class TestLoadConcurrency:
    """
    25 concurrent POST /tasks.  CoderAgent mock sleeps 1.5 s so tasks stay in
    "coding" status long enough for a genuine mid-flight DELETE to land.

    Verifies:
      - Semaphore cap respected (peak active sandbox ≤ max_concurrent_tasks=5)
      - No deadlocks — all 30 tasks reach a terminal state
      - 4 tasks cancelled while actively running → DELETE 204, final status
        "failed", error contains "Cancelled by user"
      - Cancelling one task does NOT affect the others (isolation)
      - Cancelled tasks do NOT change status again after DELETE returns
      - GET /tasks/{id} responds quickly throughout load
    """

    async def test_load_25_concurrent(self, fresh_app, monkeypatch) -> None:
        # ── Concurrency counter — thread-safe (sandbox runs in thread pool) ──
        peak_concurrent = 0
        current_concurrent = 0
        concurrent_lock = threading.Lock()

        # ── Slow coder mock: 1.5 s pause so tasks stay in "coding" long enough
        #    for the mid-flight DELETE to arrive before they complete.
        async def slow_complete(*args, **kwargs):
            await asyncio.sleep(1.5)
            return _CODER_V1

        def counting_sandbox(self, artifacts: list) -> TestResult:
            nonlocal peak_concurrent, current_concurrent
            with concurrent_lock:
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent
            time.sleep(0.05)
            with concurrent_lock:
                current_concurrent -= 1
            return SANDBOX_PASS

        monkeypatch.setattr(
            IntelligenceRouter, "complete", AsyncMock(return_value=_COMPLEXITY_SIMPLE)
        )
        monkeypatch.setattr(CoderAgent, "complete", AsyncMock(side_effect=slow_complete))
        monkeypatch.setattr(DockerSandbox, "_run_sync", counting_sandbox)

        from atlas.config import settings
        semaphore_limit = settings.max_concurrent_tasks

        # Active statuses: task has acquired the semaphore and is inside the pipeline
        active_statuses = {"planning", "coding", "verifying", "retry"}

        load_start = time.monotonic()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fresh_app), base_url="http://test"
        ) as client:

            # ── PHASE 1: Fire 25 tasks concurrently ──────────────────────
            responses = await asyncio.gather(
                *[client.post("/tasks", json=TASK_BODY) for _ in range(25)]
            )
            initial_ids = [r.json()["task_id"] for r in responses]
            assert all(r.status_code == 202 for r in responses), "Not all POSTs returned 202"

            # ── PHASE 2: Cancel 4 tasks that are actively running ─────────
            # For each target, spin-poll until the task has entered the pipeline
            # (status != "pending"), then issue DELETE mid-flight.
            cancel_ids = initial_ids[:4]
            cancel_results: dict[str, int] = {}
            statuses_before_cancel: dict[str, str] = {}

            for tid in cancel_ids:
                deadline = time.monotonic() + 10.0
                status = "pending"
                while time.monotonic() < deadline:
                    r = await client.get(f"/tasks/{tid}")
                    status = r.json()["status"]
                    if status in active_statuses or status in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.01)
                statuses_before_cancel[tid] = status

                r = await client.delete(f"/tasks/{tid}")
                cancel_results[tid] = r.status_code

            # ── PHASE 3: Poll a sample while other tasks are still running ─
            sample_get_times: list[float] = []
            for tid in initial_ids[5:10]:
                t0 = time.monotonic()
                r = await client.get(f"/tasks/{tid}")
                elapsed_ms = (time.monotonic() - t0) * 1000
                sample_get_times.append(elapsed_ms)
                assert r.status_code == 200

            # ── PHASE 4: Add 5 more tasks mid-load ───────────────────────
            mid_responses = await asyncio.gather(
                *[client.post("/tasks", json=TASK_BODY) for _ in range(5)]
            )
            mid_ids = [r.json()["task_id"] for r in mid_responses]
            assert all(r.status_code == 202 for r in mid_responses)

            # ── PHASE 5: Wait for ALL tasks to reach a terminal state ─────
            all_ids = initial_ids + mid_ids
            terminal_results: dict[str, dict] = {}

            wait_coros = [
                wait_terminal(client, tid, timeout=60.0)
                for tid in all_ids
            ]
            gathered = await asyncio.gather(*wait_coros, return_exceptions=True)

            for tid, result in zip(all_ids, gathered):
                if isinstance(result, Exception):
                    terminal_results[tid] = {"status": "TIMEOUT", "error": str(result)}
                else:
                    terminal_results[tid] = result

            # ── PHASE 6: Final list integrity check ───────────────────────
            t0 = time.monotonic()
            list_r = await client.get("/tasks")
            list_latency_ms = (time.monotonic() - t0) * 1000

            # ── PHASE 7: Post-cancel stability — re-fetch cancelled tasks ──
            # 100 ms after DELETE, confirm status is still exactly "failed" /
            # "Cancelled by user" (no pipeline except-block overwriting it).
            await asyncio.sleep(0.1)
            post_cancel_states: dict[str, tuple] = {}
            for tid in cancel_ids:
                r = await client.get(f"/tasks/{tid}")
                d = r.json()
                post_cancel_states[tid] = (d.get("status"), d.get("error"))

        total_duration = time.monotonic() - load_start

        # ── ANALYSIS ─────────────────────────────────────────────────────
        statuses = {tid: d.get("status") for tid, d in terminal_results.items()}
        completed_count = sum(1 for s in statuses.values() if s == "completed")
        failed_count = sum(1 for s in statuses.values() if s == "failed")
        timeout_count = sum(1 for s in statuses.values() if s == "TIMEOUT")

        # Strict cancellation check: must be DELETE=204, final status=failed,
        # and error must contain "Cancelled by user".
        cancelled_correctly = 0
        cancellation_errors: list[str] = []
        for tid in cancel_ids:
            http_code = cancel_results.get(tid, -1)
            final = terminal_results.get(tid, {})
            f_status = final.get("status")
            f_error = final.get("error", "")
            if http_code == 204 and f_status == "failed" and "Cancelled by user" in f_error:
                cancelled_correctly += 1
            else:
                cancellation_errors.append(
                    f"{tid}: DELETE={http_code}, status={f_status!r}, error={f_error!r}"
                )

        # Non-cancelled tasks should all reach "completed"
        non_cancel_ids = [tid for tid in all_ids if tid not in cancel_ids]
        non_cancel_completed = sum(
            1 for tid in non_cancel_ids
            if terminal_results.get(tid, {}).get("status") == "completed"
        )

        # ── REPORT ───────────────────────────────────────────────────────
        cap_ok = "YES" if peak_concurrent <= semaphore_limit else f"NO (peak={peak_concurrent})"
        print(f"""
+========================================================+
|         LOAD TEST REPORT (with mid-flight cancel)      |
+========================================================+

Tasks submitted:       {len(all_ids)} (25 initial + 5 mid-load)
Cancel targets:        {len(cancel_ids)} (first 4 task IDs)
Statuses at cancel:    {statuses_before_cancel}
Cancel HTTP results:   {cancel_results}

FINAL STATUS COUNTS
  completed:  {completed_count}   (expected {len(all_ids) - len(cancel_ids)})
  failed:     {failed_count}    (expected {len(cancel_ids)} -- all from cancel)
  TIMEOUT:    {timeout_count}

CANCELLATION DETAIL
  Cancelled correctly (DELETE=204, status=failed, "Cancelled by user"):
    {cancelled_correctly}/{len(cancel_ids)}
  Errors:
    {cancellation_errors or "none"}
  Post-cancel stability (status, error re-fetched after 100 ms):
    {post_cancel_states}

NON-CANCELLED TASK ISOLATION
  Non-cancelled tasks: {len(non_cancel_ids)}
  Reached "completed": {non_cancel_completed}
  Isolation OK:        {"YES" if non_cancel_completed == len(non_cancel_ids) else "NO -- some affected!"}

CONCURRENCY (B4)
  Semaphore cap:           {semaphore_limit}
  Peak concurrent sandbox: {peak_concurrent}
  Cap respected:           {cap_ok}

SERVER RESPONSIVENESS
  Mid-load GET /tasks/{{id}} latencies: {[f"{t:.1f}ms" for t in sample_get_times]}
  Final GET /tasks latency:            {list_latency_ms:.1f}ms
  GET stalls (>500ms):                 {sum(1 for t in sample_get_times if t > 500)}

TIMING
  Total load test duration: {total_duration:.2f}s
  Avg per task:             {total_duration / len(all_ids) * 1000:.0f}ms
+========================================================+""")

        # ── ASSERTIONS ───────────────────────────────────────────────────
        assert timeout_count == 0, (
            f"DEADLOCK: {timeout_count} tasks timed out: "
            + str([tid for tid, d in terminal_results.items() if d.get("status") == "TIMEOUT"])
        )

        assert completed_count + failed_count == len(all_ids), (
            f"Not all tasks reached terminal state: "
            f"completed={completed_count} + failed={failed_count} != {len(all_ids)}"
        )

        assert cancelled_correctly == len(cancel_ids), (
            f"Mid-flight cancellation incomplete: {cancelled_correctly}/{len(cancel_ids)} correct. "
            f"Errors: {cancellation_errors}"
        )

        assert non_cancel_completed == len(non_cancel_ids), (
            f"Cancelling tasks leaked into others: "
            f"{non_cancel_completed}/{len(non_cancel_ids)} non-cancelled tasks completed"
        )

        assert peak_concurrent <= semaphore_limit, (
            f"B4 VIOLATED: peak concurrent sandbox={peak_concurrent} > semaphore={semaphore_limit}"
        )

        # Cancelled tasks must stay stable after DELETE (no pipeline overwrite)
        for tid in cancel_ids:
            stable_status, stable_error = post_cancel_states.get(tid, (None, None))
            assert stable_status == "failed", (
                f"Task {tid} status drifted after cancel: {stable_status!r}"
            )
            assert "Cancelled by user" in (stable_error or ""), (
                f"Task {tid} error changed after cancel: {stable_error!r}"
            )

        assert max(sample_get_times, default=0) < 2000, (
            f"GET /tasks stalled during load: max latency {max(sample_get_times):.1f}ms"
        )

        assert list_r.status_code == 200
