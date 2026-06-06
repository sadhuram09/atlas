"""
atlas/contracts_v3.py

Phase 3 extensions — Memory + Governor contracts.

New concepts:
  - FailurePattern: a stored bug + fix pair in the FAISS index
  - MemorySearchResult: what comes back from a similarity search
  - BudgetState: real-time cost tracking per task
  - GovernorDecision: model routing decision with reasoning
  - EnhancedPrompt: a prompt enriched with relevant past failure patterns
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from atlas.contracts import ATLASModel, AgentRole, ModelTier


# ---------------------------------------------------------------------------
# Failure memory
# ---------------------------------------------------------------------------

class FailurePattern(ATLASModel):
    """
    A stored failure + fix pair in the FAISS index.

    When a subtask fails and then passes on retry, we store:
      - The error message that caused the failure
      - The code that was wrong
      - The code that fixed it
      - What kind of error it was

    On future tasks, we search for similar errors and surface this pattern
    to CoderAgent before it retries — giving it the exact fix that worked.
    """
    id: str
    task_id: str
    subtask_title: str
    language: str
    error_type: str          # e.g. "AssertionError", "TypeError", "ImportError"
    error_summary: str       # First 500 chars of the failure output
    failed_code: str         # The code that failed
    fixed_code: str          # The code that passed
    fix_description: str     # One-sentence description of what changed
    test_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemorySearchResult(ATLASModel):
    """Result of a FAISS similarity search."""
    pattern: FailurePattern
    similarity_score: float  # 0.0 = unrelated, 1.0 = identical
    rank: int                # 1 = most similar


class MemoryInsight(ATLASModel):
    """
    Processed memory results ready to inject into a prompt.

    The PromptEnhancer converts MemorySearchResults into this —
    a clean, concise description of relevant past failures and fixes.
    """
    relevant_patterns: list[MemorySearchResult]
    prompt_addition: str     # The actual text injected into the prompt
    patterns_found: int
    most_similar_score: float


# ---------------------------------------------------------------------------
# Governor — budget-aware model routing
# ---------------------------------------------------------------------------

class BudgetState(ATLASModel):
    """
    Real-time cost tracking for a task.

    The Governor checks this before every LLM call and decides
    whether to downgrade the model tier to stay within budget.
    """
    task_id: str
    budget_usd: float                    # User-set limit
    spent_usd: float = 0.0              # Running total
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tier_downgrades: int = 0            # How many times we dropped a tier

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    @property
    def percent_used(self) -> float:
        if self.budget_usd <= 0:
            return 100.0
        return (self.spent_usd / self.budget_usd) * 100

    @property
    def is_critical(self) -> bool:
        """True if over 80% of budget consumed."""
        return self.percent_used >= 80.0

    @property
    def is_exhausted(self) -> bool:
        return self.spent_usd >= self.budget_usd


class GovernorDecision(ATLASModel):
    """
    A model routing decision made by the Governor.

    The Governor considers: task complexity, budget remaining,
    subtask position in DAG, and historical cost per tier.
    """
    requested_tier: ModelTier
    approved_tier: ModelTier       # May be downgraded if budget is low
    model: str
    reasoning: str
    budget_state: BudgetState
    downgraded: bool = False       # True if tier was lowered


# ---------------------------------------------------------------------------
# Enhanced prompt
# ---------------------------------------------------------------------------

class EnhancedPrompt(ATLASModel):
    """
    A prompt enriched with memory context.

    The original prompt + relevant past failure patterns injected
    before CoderAgent makes its LLM call.
    """
    original_prompt: str
    memory_context: str          # Injected failure patterns
    enhanced_prompt: str         # original + memory_context combined
    patterns_used: int
    enhancement_applied: bool
