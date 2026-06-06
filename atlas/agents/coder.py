"""
atlas/agents/coder.py

CoderAgent — writes production-quality code given a task description.

Flow:
  1. Receives AgentMessage with task title + description + language
  2. Calls Groq (llama-3.3-70b-versatile) with a precise system prompt
  3. Parses the response to extract clean code blocks
  4. Returns an AgentMessage with a list of Artifact objects

On retry (called again after a failed VerifierAgent run):
  - Receives the previous code + the test failure output
  - Groq sees exactly what went wrong and fixes it
  - This is the "self-healing" part of the critic loop

Design decisions:
  - Uses BALANCED tier (llama-3.3-70b) — best code quality on Groq free tier
  - temperature=0.1 — we want deterministic, precise code, not creativity
  - Extracts ALL code blocks from the response — handles multi-file output
  - Always generates a test file alongside the implementation
"""

from __future__ import annotations

import hashlib
import re

from atlas.agents.base import BaseAgent, AgentError
from atlas.contracts import (
    AgentMessage,
    AgentRole,
    Artifact,
    ModelTier,
)
from atlas.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt — the instruction set that shapes every code response
# ---------------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """You are ATLAS CoderAgent, an expert software engineer.

Your job: write clean, correct, production-quality code.

STRICT OUTPUT FORMAT — you must always respond with exactly this structure:

## Implementation

```{language}
# Full implementation here
```

## Tests

```{language}
# pytest tests here — test every function and edge case
# Always import from the implementation file
# Always use def test_* naming
```

## Explanation
One paragraph explaining what you built and why.

RULES:
- Code must be complete and runnable — no placeholders, no "..." 
- Tests must use pytest and actually import + call the implementation
- Include type hints on all functions
- Handle edge cases explicitly
- No external dependencies beyond the Python standard library unless specified
- The implementation filename will be solution.py
- The test filename will be test_solution.py
"""


class CoderAgent(BaseAgent):
    """
    Writes code and generates tests for a given task.

    On first attempt: writes from scratch.
    On retry: receives previous code + failure output and fixes the bugs.
    """

    role = AgentRole.CODER

    async def run(self, message: AgentMessage) -> AgentMessage:
        """
        Generate code artifacts from a task description.

        Payload in:
            task_title: str
            task_description: str
            language: str
            attempt: int          (0 = first try, 1+ = retry)
            previous_code: str    (only on retry)
            test_failure: str     (only on retry — the pytest output)

        Payload out:
            artifacts: list[dict]   (serialised Artifact objects)
            explanation: str
            attempt: int
        """
        payload = message.payload
        title = payload["task_title"]
        description = payload["task_description"]
        language = payload.get("language", "python")
        attempt = payload.get("attempt", 0)
        previous_code = payload.get("previous_code", "")
        test_failure = payload.get("test_failure", "")

        self.log.info(
            "coder_started",
            title=title,
            language=language,
            attempt=attempt,
        )

        # Build the prompt — different on retry vs first attempt
        if attempt == 0:
            prompt = self._build_initial_prompt(title, description, language)
        else:
            prompt = self._build_retry_prompt(
                title, description, language,
                previous_code, test_failure, attempt
            )

        # Call Groq — BALANCED tier = llama-3.3-70b-versatile
        system = CODER_SYSTEM_PROMPT.replace("{language}", language)
        raw_response = await self.complete(
            prompt=prompt,
            system=system,
            tier=ModelTier.BALANCED,
            max_tokens=4096,
            temperature=0.1,  # Low temp = precise, deterministic code
        )

        # Parse the response into clean Artifact objects
        artifacts = self._extract_artifacts(raw_response, language)
        explanation = self._extract_explanation(raw_response)

        if not artifacts:
            raise AgentError(
                f"CoderAgent produced no code blocks on attempt {attempt}",
                agent=self.role,
                task_id=self.task_id,
            )

        self.log.info(
            "coder_completed",
            artifacts=len(artifacts),
            attempt=attempt,
            total_tokens=self._total_tokens,
        )

        return self._reply(
            message,
            payload={
                "artifacts": [a.model_dump() for a in artifacts],
                "explanation": explanation,
                "attempt": attempt,
            },
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_initial_prompt(
        self, title: str, description: str, language: str
    ) -> str:
        return f"""Task: {title}

Description:
{description}

Language: {language}

Write the complete implementation and tests now."""

    def _build_retry_prompt(
        self,
        title: str,
        description: str,
        language: str,
        previous_code: str,
        test_failure: str,
        attempt: int,
    ) -> str:
        return f"""Task: {title}

Description:
{description}

Language: {language}

RETRY ATTEMPT {attempt} — Your previous code failed the tests.

Previous code:
```{language}
{previous_code}
```

Test failure output:
```
{test_failure}
```

Study the failure carefully. Fix the bugs and rewrite both the implementation
and the tests. Make sure all tests pass this time."""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _extract_artifacts(self, response: str, language: str) -> list[Artifact]:
        """
        Extract all fenced code blocks from the LLM response.

        The system prompt enforces:
          Block 1 → solution.{ext}      (implementation)
          Block 2 → test_solution.{ext} (tests)

        We handle cases where the model forgets the fence language tag.
        """
        ext = self._extension(language)

        # Match ```language ... ``` or just ``` ... ```
        pattern = re.compile(
            r"```(?:\w+)?\n(.*?)```",
            re.DOTALL,
        )
        blocks = pattern.findall(response)

        if not blocks:
            return []

        filenames = [f"solution.{ext}", f"test_solution.{ext}"]
        artifacts: list[Artifact] = []

        for i, block in enumerate(blocks[:2]):  # Max 2 files
            content = block.strip()
            if not content:
                continue

            filename = filenames[i] if i < len(filenames) else f"file_{i}.{ext}"
            checksum = hashlib.sha256(content.encode()).hexdigest()

            artifacts.append(
                Artifact(
                    filename=filename,
                    language=language,
                    content=content,
                    checksum=checksum,
                )
            )

        return artifacts

    def _extract_explanation(self, response: str) -> str:
        """Pull the explanation paragraph from the response."""
        # Look for the ## Explanation section
        match = re.search(
            r"##\s*Explanation\s*\n(.*?)(?:\n##|\Z)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extension(language: str) -> str:
        """Map language name to file extension."""
        return {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "go": "go",
            "rust": "rs",
            "java": "java",
        }.get(language.lower(), "py")
