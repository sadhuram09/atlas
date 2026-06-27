"""
tests/unit/test_sandbox.py

Tests for DockerSandbox — specifically the output parsing logic.

These tests don't require Docker to be running.
They test _parse_pytest_output directly with realistic pytest output strings.
"""

import pytest
from atlas.tools.sandbox import DockerSandbox
from atlas.contracts import Artifact


@pytest.fixture
def sandbox() -> DockerSandbox:
    return DockerSandbox()


PASSING_OUTPUT = """
test_solution.py::test_fibonacci_zero PASSED
test_solution.py::test_fibonacci_one PASSED
test_solution.py::test_fibonacci_ten PASSED
test_solution.py::test_negative_raises PASSED

4 passed in 0.12s
"""

FAILING_OUTPUT = """
test_solution.py::test_fibonacci_zero PASSED
test_solution.py::test_fibonacci_ten FAILED

FAILURES
--------
FAILED test_solution.py::test_fibonacci_ten - AssertionError: assert 54 == 55
FAILED test_solution.py::test_negative_raises - ValueError not raised

2 failed, 1 passed in 0.08s
"""

NO_TESTS_OUTPUT = """
collected 0 items

no tests ran
"""


class TestPytestOutputParsing:
    def test_passing_output_is_marked_passed(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(PASSING_OUTPUT, "", 0, 120)
        assert result.passed is True

    def test_failing_output_is_marked_failed(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(FAILING_OUTPUT, "", 1, 80)
        assert result.passed is False

    def test_exit_code_stored(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(PASSING_OUTPUT, "", 0, 100)
        assert result.exit_code == 0

    def test_passing_test_count(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(PASSING_OUTPUT, "", 0, 100)
        assert result.test_count == 4

    def test_failing_test_count_includes_both(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(FAILING_OUTPUT, "", 1, 100)
        assert result.test_count == 3  # 2 failed + 1 passed

    def test_failed_test_names_extracted(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(FAILING_OUTPUT, "", 1, 100)
        assert "test_fibonacci_ten" in result.failed_tests
        assert "test_negative_raises" in result.failed_tests

    def test_no_failed_tests_on_pass(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(PASSING_OUTPUT, "", 0, 100)
        assert result.failed_tests == []

    def test_duration_ms_stored(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(PASSING_OUTPUT, "", 0, 250)
        assert result.duration_ms == 250

    def test_stdout_stored(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output(PASSING_OUTPUT, "", 0, 100)
        assert "PASSED" in result.stdout

    def test_stderr_stored(self, sandbox: DockerSandbox) -> None:
        result = sandbox._parse_pytest_output("", "some warning", 0, 100)
        assert "some warning" in result.stderr


class TestSandboxEmptyArtifacts:
    async def test_empty_artifacts_returns_failed_result(self, sandbox: DockerSandbox) -> None:
        result = await sandbox.run([])
        assert result.passed is False
        assert "No artifacts" in result.stderr

    async def test_no_test_file_returns_failed(self, sandbox: DockerSandbox) -> None:
        artifacts = [
            Artifact(
                filename="solution.py",
                language="python",
                content="def add(a, b): return a + b",
            )
        ]
        result = await sandbox.run(artifacts)
        assert result.passed is False
        assert "test file" in result.stderr.lower()
