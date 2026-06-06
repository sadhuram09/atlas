"""
tests/unit/test_coder.py

Tests for CoderAgent — specifically the parsing and extraction logic.

These tests don't call Groq. They test the _extract_artifacts and
_build_prompt methods directly, which is where most bugs hide.

Pattern: feed the agent a realistic LLM response string,
assert it extracts the right code blocks into the right filenames.
"""

import pytest
from atlas.agents.coder import CoderAgent


# A realistic response the LLM would produce
SAMPLE_LLM_RESPONSE = """
## Implementation

```python
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number using dynamic programming.\"\"\"
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

## Tests

```python
import pytest
from solution import fibonacci

def test_fibonacci_zero():
    assert fibonacci(0) == 0

def test_fibonacci_one():
    assert fibonacci(1) == 1

def test_fibonacci_ten():
    assert fibonacci(10) == 55

def test_fibonacci_negative_raises():
    with pytest.raises(ValueError):
        fibonacci(-1)
```

## Explanation
I implemented fibonacci using bottom-up dynamic programming for O(n) time
and O(n) space complexity, avoiding the exponential cost of naive recursion.
"""

RESPONSE_NO_TESTS = """
## Implementation

```python
def add(a: int, b: int) -> int:
    return a + b
```

## Explanation
Simple addition function.
"""


@pytest.fixture
def coder() -> CoderAgent:
    return CoderAgent(task_id="test-task-001")


class TestArtifactExtraction:
    def test_extracts_two_files(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(SAMPLE_LLM_RESPONSE, "python")
        assert len(artifacts) == 2

    def test_implementation_filename(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(SAMPLE_LLM_RESPONSE, "python")
        filenames = [a.filename for a in artifacts]
        assert "solution.py" in filenames

    def test_test_filename(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(SAMPLE_LLM_RESPONSE, "python")
        filenames = [a.filename for a in artifacts]
        assert "test_solution.py" in filenames

    def test_implementation_contains_function(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(SAMPLE_LLM_RESPONSE, "python")
        impl = next(a for a in artifacts if a.filename == "solution.py")
        assert "def fibonacci" in impl.content

    def test_test_contains_test_functions(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(SAMPLE_LLM_RESPONSE, "python")
        tests = next(a for a in artifacts if a.filename == "test_solution.py")
        assert "def test_" in tests.content

    def test_artifacts_have_checksums(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(SAMPLE_LLM_RESPONSE, "python")
        for a in artifacts:
            assert len(a.checksum) == 64  # SHA-256 hex = 64 chars

    def test_artifacts_have_correct_language(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(SAMPLE_LLM_RESPONSE, "python")
        for a in artifacts:
            assert a.language == "python"

    def test_empty_response_returns_empty(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts("no code blocks here", "python")
        assert artifacts == []

    def test_single_block_response(self, coder: CoderAgent) -> None:
        artifacts = coder._extract_artifacts(RESPONSE_NO_TESTS, "python")
        assert len(artifacts) == 1
        assert artifacts[0].filename == "solution.py"


class TestExplanationExtraction:
    def test_extracts_explanation(self, coder: CoderAgent) -> None:
        explanation = coder._extract_explanation(SAMPLE_LLM_RESPONSE)
        assert "dynamic programming" in explanation

    def test_no_explanation_returns_empty(self, coder: CoderAgent) -> None:
        explanation = coder._extract_explanation("```python\npass\n```")
        assert explanation == ""


class TestPromptBuilding:
    def test_initial_prompt_contains_title(self, coder: CoderAgent) -> None:
        prompt = coder._build_initial_prompt(
            "Sort function", "Write a merge sort", "python"
        )
        assert "Sort function" in prompt

    def test_initial_prompt_contains_description(self, coder: CoderAgent) -> None:
        prompt = coder._build_initial_prompt(
            "Sort function", "Write a merge sort", "python"
        )
        assert "Write a merge sort" in prompt

    def test_retry_prompt_contains_failure(self, coder: CoderAgent) -> None:
        prompt = coder._build_retry_prompt(
            "Sort", "Write sort", "python",
            "def sort(): pass",
            "FAILED test_solution.py::test_sort",
            attempt=1,
        )
        assert "FAILED" in prompt
        assert "RETRY ATTEMPT 1" in prompt

    def test_retry_prompt_contains_previous_code(self, coder: CoderAgent) -> None:
        prompt = coder._build_retry_prompt(
            "Sort", "Write sort", "python",
            "def sort(): pass",
            "test failed",
            attempt=1,
        )
        assert "def sort(): pass" in prompt


class TestFileExtensions:
    def test_python_extension(self, coder: CoderAgent) -> None:
        assert coder._extension("python") == "py"

    def test_javascript_extension(self, coder: CoderAgent) -> None:
        assert coder._extension("javascript") == "js"

    def test_unknown_defaults_to_py(self, coder: CoderAgent) -> None:
        assert coder._extension("brainfuck") == "py"
