"""
atlas/tools/sandbox.py

DockerSandbox — runs generated code in a fully isolated Docker container.

Why Docker?
  We're executing AI-generated code. It could do anything:
  - Delete files
  - Make network requests  
  - Eat all your CPU
  - Import malicious packages

  Docker gives us:
  - Filesystem isolation (can't touch host files)
  - Network isolation (network="none" — no internet)
  - Memory limits (256MB max)
  - CPU limits
  - Automatic cleanup after each run

How it works:
  1. Write artifacts to a temp directory on the host
  2. Mount that directory into a python:3.12-slim container (read-only)
  3. Run `pytest test_solution.py -v` inside the container
  4. Capture stdout/stderr + exit code
  5. Parse the pytest output into a TestResult
  6. Delete the container and temp files

The container never touches your project files.
It runs for a maximum of `docker_timeout_seconds` (default: 30s).

Windows note:
  Docker Desktop must be running. Install from docker.com/products/docker-desktop
  The sandbox works on Windows via Docker Desktop's WSL2 backend.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from atlas.contracts import Artifact, TestResult
from atlas.config import settings
from atlas.logging import get_logger

log = get_logger(__name__)


class SandboxError(Exception):
    """Raised when Docker itself fails (not when tests fail)."""


class DockerSandbox:
    """
    Runs code artifacts in an isolated Docker container.

    Usage:
        sandbox = DockerSandbox()
        available = await sandbox.is_available()

        result = await sandbox.run(artifacts)
        if result.passed:
            print("All tests passed!")
        else:
            print(result.stdout)  # pytest output with failure details
    """

    # The Docker image used for execution.
    # python:3.12-slim is official, trusted, and has pytest pre-installable.
    IMAGE = "python:3.12-slim"

    def __init__(self) -> None:
        self.timeout = settings.docker_timeout_seconds
        self.memory_limit = settings.docker_memory_limit

    def is_available(self) -> bool:
        """
        Check if Docker is running and reachable.

        Called at startup — if Docker is unavailable, the sandbox
        falls back to direct subprocess execution with a warning.
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(self, artifacts: list[Artifact]) -> TestResult:
        """
        Write artifacts to a temp dir and run pytest in Docker.

        Args:
            artifacts: List of Artifact objects (solution.py + test_solution.py)

        Returns:
            TestResult with passed/failed status and full pytest output.

        Raises:
            SandboxError: if Docker itself crashes (not if tests fail)
        """
        if not artifacts:
            return TestResult(
                passed=False,
                exit_code=1,
                stdout="",
                stderr="No artifacts to test",
                duration_ms=0,
            )

        # Find the test file — always starts with "test_"
        test_file = next(
            (a for a in artifacts if a.filename.startswith("test_")),
            None,
        )
        if not test_file:
            return TestResult(
                passed=False,
                exit_code=1,
                stdout="",
                stderr="No test file found. Expected a file starting with 'test_'",
                duration_ms=0,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write all artifacts to the temp directory
            for artifact in artifacts:
                (tmp / artifact.filename).write_text(
                    artifact.content, encoding="utf-8"
                )

            log.info(
                "sandbox_run_start",
                files=[a.filename for a in artifacts],
                test_file=test_file.filename,
                timeout=self.timeout,
            )

            start = time.monotonic()

            # Try Docker first, fall back to direct subprocess if unavailable
            if self.is_available():
                result = self._run_in_docker(tmp, test_file.filename)
            else:
                log.warning(
                    "docker_unavailable",
                    message="Docker not running — using direct subprocess. "
                            "Install Docker Desktop for full isolation.",
                )
                result = self._run_direct(tmp, test_file.filename)

            duration_ms = int((time.monotonic() - start) * 1000)

        test_result = self._parse_pytest_output(
            stdout=result["stdout"],
            stderr=result["stderr"],
            exit_code=result["exit_code"],
            duration_ms=duration_ms,
        )

        log.info(
            "sandbox_run_complete",
            passed=test_result.passed,
            exit_code=test_result.exit_code,
            duration_ms=duration_ms,
            test_count=test_result.test_count,
            failed_tests=test_result.failed_tests,
        )

        return test_result

    def _run_in_docker(self, tmpdir: Path, test_filename: str) -> dict:
        """Run pytest inside a Docker container."""
        # Convert Windows path to Docker-compatible format if needed
        mount_path = str(tmpdir)
        if sys.platform == "win32":
            # Convert C:\Users\... to /c/Users/... for Docker on Windows
            mount_path = mount_path.replace("\\", "/")
            if mount_path[1] == ":":
                mount_path = "/" + mount_path[0].lower() + mount_path[2:]

        cmd = [
            "docker", "run",
            "--rm",                          # Delete container after run
            "--network", "none",             # No internet access
            f"--memory={self.memory_limit}", # Memory cap
            "--cpus=1",                      # CPU cap
            "-v", f"{mount_path}:/code:ro",  # Mount code read-only
            "-w", "/code",                   # Working directory
            self.IMAGE,
            "sh", "-c",
            f"pip install pytest -q --no-cache-dir 2>/dev/null && "
            f"python -m pytest {test_filename} -v --tb=short --no-header 2>&1",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 30,  # Extra 30s for pip install
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Docker container timed out after {self.timeout}s",
                "exit_code": 124,
            }
        except Exception as e:
            raise SandboxError(f"Docker run failed: {e}") from e

    def _run_direct(self, tmpdir: Path, test_filename: str) -> dict:
        """
        Fallback: run pytest directly in a subprocess (no Docker isolation).

        Used when Docker Desktop is not installed/running.
        Safe enough for development — use Docker in production.
        """
        cmd = [
            sys.executable, "-m", "pytest",
            str(tmpdir / test_filename),
            "-v", "--tb=short", "--no-header",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(tmpdir),
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Tests timed out after {self.timeout}s",
                "exit_code": 124,
            }

    @staticmethod
    def _parse_pytest_output(
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_ms: int,
    ) -> TestResult:
        """
        Parse pytest output into a structured TestResult.

        pytest exit codes:
          0 = all tests passed
          1 = some tests failed
          2 = interrupted (e.g. keyboard)
          3 = internal error
          4 = usage error
          5 = no tests collected
        """
        passed = exit_code == 0
        combined = stdout + "\n" + stderr

        # Count tests — look for "X passed" or "X failed" in summary line
        test_count = 0
        count_match = re.search(r"(\d+) passed", combined)
        if count_match:
            test_count = int(count_match.group(1))

        failed_match = re.search(r"(\d+) failed", combined)
        failed_count = int(failed_match.group(1)) if failed_match else 0
        test_count += failed_count

        # Extract names of failed tests
        # pytest prints: "FAILED test_solution.py::test_name - AssertionError"
        failed_tests = re.findall(
            r"FAILED\s+[\w./]+::(\w+)",
            combined,
        )

        return TestResult(
            passed=passed,
            exit_code=exit_code,
            stdout=stdout[:8000],   # Cap at 8KB to avoid huge payloads
            stderr=stderr[:2000],
            duration_ms=duration_ms,
            test_count=test_count,
            failed_tests=failed_tests,
        )
