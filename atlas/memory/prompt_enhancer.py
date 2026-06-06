"""
atlas/memory/prompt_enhancer.py

PromptEnhancer — enriches retry prompts with relevant past failure patterns.

This is the bridge between FailureMemory and CoderAgent.

On a retry:
  1. CoderAgent has: the original task + the test failure output
  2. PromptEnhancer searches FailureMemory for similar past failures
  3. If found: injects "here's how we fixed a similar bug before" into the prompt
  4. CoderAgent now sees: task + failure + relevant past fix examples

Why this is powerful:
  Without memory: CoderAgent sees the error and guesses at a fix.
  With memory: CoderAgent sees the error AND 1-3 concrete examples of
  fixes that worked for similar errors. This is the difference between
  "I think the problem might be X" and "Last time we saw this exact
  AssertionError pattern, the fix was to add None checking here."

The injection is structured so the LLM understands exactly what it is:
  - Clear separation from the main prompt
  - Each pattern shows: error → what was wrong → what fixed it
  - Similarity score included so LLM can weight relevance
  - Keeps it concise — relevant context, not noise
"""

from __future__ import annotations

from atlas.contracts_v3 import EnhancedPrompt, MemoryInsight, MemorySearchResult
from atlas.memory.failure_memory import failure_memory
from atlas.logging import get_logger

log = get_logger(__name__)

# Maximum characters of memory context to inject
# Keep it tight — LLM context is finite and we don't want to drown the task
MAX_CONTEXT_CHARS = 2000


class PromptEnhancer:
    """
    Enriches prompts with relevant past failure patterns.

    Stateless — all state lives in the FailureMemory singleton.
    """

    def enhance_retry_prompt(
        self,
        original_prompt: str,
        error_output: str,
        language: str = "python",
        top_k: int = 3,
    ) -> EnhancedPrompt:
        """
        Search memory and inject relevant past fixes into the prompt.

        Args:
            original_prompt: The prompt CoderAgent would send without memory
            error_output: The pytest failure output from the last attempt
            language: Programming language (filters memory search)
            top_k: Max number of patterns to inject

        Returns:
            EnhancedPrompt with the memory context injected.
            If no relevant patterns found, returns prompt unchanged.
        """
        if not failure_memory.is_available or failure_memory.pattern_count == 0:
            return EnhancedPrompt(
                original_prompt=original_prompt,
                memory_context="",
                enhanced_prompt=original_prompt,
                patterns_used=0,
                enhancement_applied=False,
            )

        # Search for similar past failures
        results = failure_memory.search(
            error_output=error_output,
            language=language,
            top_k=top_k,
        )

        if not results:
            log.info("prompt_enhancer_no_matches", error_chars=len(error_output))
            return EnhancedPrompt(
                original_prompt=original_prompt,
                memory_context="",
                enhanced_prompt=original_prompt,
                patterns_used=0,
                enhancement_applied=False,
            )

        # Build the memory context block
        memory_context = self._build_context(results)

        # Inject it into the prompt
        enhanced_prompt = self._inject_context(original_prompt, memory_context)

        log.info(
            "prompt_enhancer_applied",
            patterns=len(results),
            top_similarity=results[0].similarity_score,
            context_chars=len(memory_context),
        )

        return EnhancedPrompt(
            original_prompt=original_prompt,
            memory_context=memory_context,
            enhanced_prompt=enhanced_prompt,
            patterns_used=len(results),
            enhancement_applied=True,
        )

    def _build_context(self, results: list[MemorySearchResult]) -> str:
        """
        Build a concise, structured memory context block.

        Format:
          ATLAS MEMORY — Similar past failures and their fixes:

          [Pattern 1] Similarity: 0.87 | Error: AssertionError
          What failed: <first 200 chars of broken code>
          What fixed it: <fix description>
          Fixed code excerpt: <first 200 chars of working code>

          [Pattern 2] ...
        """
        lines = [
            "ATLAS MEMORY — Similar failures from past tasks and how they were fixed:",
            "",
        ]

        total_chars = len(lines[0])

        for result in results:
            p = result.pattern

            block = [
                f"[Pattern {result.rank}] Similarity: {result.similarity_score:.2f} | "
                f"Error type: {p.error_type}",
                f"Past error summary: {p.error_summary[:200]}",
                f"What fixed it: {p.fix_description}",
                f"Working code excerpt:",
                f"```{p.language}",
                p.fixed_code[:300],
                "```",
                "",
            ]

            block_text = "\n".join(block)

            # Don't exceed max context size
            if total_chars + len(block_text) > MAX_CONTEXT_CHARS:
                break

            lines.extend(block)
            total_chars += len(block_text)

        lines.append("Use the above patterns as guidance — adapt, don't copy blindly.")
        return "\n".join(lines)

    @staticmethod
    def _inject_context(original_prompt: str, memory_context: str) -> str:
        """
        Inject memory context between the task description and the instruction.

        We inject BEFORE "Write the complete implementation" so the LLM
        sees the context as part of the problem framing, not as an afterthought.
        """
        injection = f"\n\n{memory_context}\n\n"

        # Inject before the final instruction line if present
        if "Write the complete implementation" in original_prompt:
            return original_prompt.replace(
                "Write the complete implementation",
                f"{injection}Write the complete implementation",
                1,
            )

        # Otherwise append at the end
        return original_prompt + injection


# Singleton
prompt_enhancer = PromptEnhancer()
