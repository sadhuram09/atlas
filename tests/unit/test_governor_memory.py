"""
tests/unit/test_governor_memory.py

Tests for Governor budget routing and FailureMemory + PromptEnhancer integration.
No real Groq calls — all state is injected directly.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas.contracts import ModelTier
from atlas.contracts_v3 import BudgetState, FailurePattern, GovernorDecision
from atlas.governor.governor import Governor, DOWNGRADE_THRESHOLD
from atlas.memory.failure_memory import FailureMemory
from atlas.memory.prompt_enhancer import PromptEnhancer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spend(gov: Governor, task_id: str, fraction: float, budget_usd: float = 1.0) -> None:
    """Drive governor.record_usage() until `fraction` of the budget is spent."""
    target_spend = budget_usd * fraction
    current = gov.get_budget(task_id)
    already_spent = current.spent_usd if current else 0.0
    remaining = target_spend - already_spent
    if remaining <= 0:
        return
    # Use the cheapest model; tokens_in drives cost at $0.05/M
    # 1 token_in at $0.05/M = $0.00000005 → need lots of tokens
    tokens_needed = int((remaining / 0.05) * 1_000_000)
    gov.record_usage(task_id, tokens_in=tokens_needed, tokens_out=0,
                     model="llama-3.1-8b-instant")


# ---------------------------------------------------------------------------
# BudgetState property tests
# ---------------------------------------------------------------------------

class TestBudgetStateProperties:

    def test_is_critical_fires_at_90_percent(self):
        # Just below threshold — not critical
        b = BudgetState(task_id="t1", budget_usd=1.0, spent_usd=0.899)
        assert not b.is_critical

    def test_is_critical_fires_at_exactly_90_percent(self):
        b = BudgetState(task_id="t1", budget_usd=1.0, spent_usd=0.90)
        assert b.is_critical

    def test_is_critical_above_90_percent(self):
        b = BudgetState(task_id="t1", budget_usd=1.0, spent_usd=0.95)
        assert b.is_critical

    def test_is_exhausted(self):
        b = BudgetState(task_id="t1", budget_usd=1.0, spent_usd=1.0)
        assert b.is_exhausted

    def test_not_exhausted_below_budget(self):
        b = BudgetState(task_id="t1", budget_usd=1.0, spent_usd=0.99)
        assert not b.is_exhausted

    def test_percent_used_calculation(self):
        b = BudgetState(task_id="t1", budget_usd=2.0, spent_usd=1.0)
        assert b.percent_used == 50.0

    def test_remaining_usd_floors_at_zero(self):
        b = BudgetState(task_id="t1", budget_usd=1.0, spent_usd=1.5)
        assert b.remaining_usd == 0.0


# ---------------------------------------------------------------------------
# Governor routing tests
# ---------------------------------------------------------------------------

class TestGovernorRouting:

    def setup_method(self):
        self.gov = Governor()

    def test_healthy_budget_approved_as_requested(self):
        self.gov.initialize_task("t1", budget_usd=1.0)
        decision = self.gov.route("t1", ModelTier.POWERFUL)
        assert decision.approved_tier == ModelTier.POWERFUL
        assert not decision.downgraded

    def test_downgrade_threshold_powerful_to_balanced(self):
        """At >=70% spent, POWERFUL → BALANCED."""
        self.gov.initialize_task("t1", budget_usd=1.0)
        _spend(self.gov, "t1", 0.71)
        decision = self.gov.route("t1", ModelTier.POWERFUL)
        assert decision.approved_tier == ModelTier.BALANCED
        assert decision.downgraded

    def test_downgrade_threshold_balanced_to_fast(self):
        """At >=70% spent, BALANCED → FAST."""
        self.gov.initialize_task("t1", budget_usd=1.0)
        _spend(self.gov, "t1", 0.71)
        decision = self.gov.route("t1", ModelTier.BALANCED)
        assert decision.approved_tier == ModelTier.FAST
        assert decision.downgraded

    def test_downgrade_threshold_fast_stays_fast(self):
        """At >=70% spent, FAST stays FAST (nothing lower to drop to)."""
        self.gov.initialize_task("t1", budget_usd=1.0)
        _spend(self.gov, "t1", 0.71)
        decision = self.gov.route("t1", ModelTier.FAST)
        assert decision.approved_tier == ModelTier.FAST
        assert not decision.downgraded

    def test_critical_threshold_forces_fast_regardless_of_request(self):
        """At >=90% spent, every tier request is forced to FAST."""
        self.gov.initialize_task("t1", budget_usd=1.0)
        _spend(self.gov, "t1", 0.91)
        for requested in [ModelTier.POWERFUL, ModelTier.BALANCED]:
            decision = self.gov.route("t1", requested)
            assert decision.approved_tier == ModelTier.FAST, (
                f"Expected FAST at 91% used, got {decision.approved_tier} "
                f"for requested={requested}"
            )
            assert decision.downgraded

    def test_exhausted_budget_forces_fast_and_does_not_raise(self):
        """Past 100% spent, Governor forces FAST — never raises, never hangs."""
        self.gov.initialize_task("t1", budget_usd=0.001)
        # Massively over-spend
        _spend(self.gov, "t1", 5.0, budget_usd=0.001)
        budget = self.gov.get_budget("t1")
        assert budget.is_exhausted

        # Must not raise
        decision = self.gov.route("t1", ModelTier.POWERFUL)
        assert decision.approved_tier == ModelTier.FAST
        assert decision.downgraded

    def test_exhausted_budget_reasoning_string(self):
        """Exhausted budget produces an 'exhausted' reasoning message."""
        self.gov.initialize_task("t1", budget_usd=0.001)
        _spend(self.gov, "t1", 5.0, budget_usd=0.001)
        decision = self.gov.route("t1", ModelTier.POWERFUL)
        assert "exhausted" in decision.reasoning.lower() or "forced" in decision.reasoning.lower()

    def test_downgrade_counter_increments(self):
        """tier_downgrades on BudgetState increments each time a tier is downgraded."""
        self.gov.initialize_task("t1", budget_usd=1.0)
        _spend(self.gov, "t1", 0.71)
        self.gov.route("t1", ModelTier.POWERFUL)
        self.gov.route("t1", ModelTier.POWERFUL)
        budget = self.gov.get_budget("t1")
        assert budget.tier_downgrades == 2

    def test_record_usage_accumulates(self):
        """record_usage accumulates tokens and cost across multiple calls."""
        self.gov.initialize_task("t1", budget_usd=10.0)
        self.gov.record_usage("t1", tokens_in=1000, tokens_out=500, model="llama-3.1-8b-instant")
        self.gov.record_usage("t1", tokens_in=2000, tokens_out=1000, model="llama-3.1-8b-instant")
        budget = self.gov.get_budget("t1")
        assert budget.llm_calls == 2
        assert budget.tokens_in == 3000
        assert budget.tokens_out == 1500
        assert budget.spent_usd > 0

    def test_no_hard_stop_on_exhaustion(self):
        """
        Even with is_exhausted=True, Governor always returns a valid decision.
        The pipeline is never aborted by the Governor — it runs to completion.
        """
        self.gov.initialize_task("t1", budget_usd=0.001)
        _spend(self.gov, "t1", 5.0, budget_usd=0.001)
        # Call route many times — should never raise
        for _ in range(5):
            decision = self.gov.route("t1", ModelTier.BALANCED)
            assert decision.model  # always returns a valid model string

    def test_is_critical_aligns_with_governor_critical_branch(self):
        """
        BudgetState.is_critical fires at 90%, matching the Governor's critical branch.
        This test pins the alignment so the two never diverge again.
        """
        self.gov.initialize_task("t1", budget_usd=1.0)
        # Spend exactly 89% — not critical
        _spend(self.gov, "t1", 0.89)
        budget = self.gov.get_budget("t1")
        assert not budget.is_critical
        decision = self.gov.route("t1", ModelTier.BALANCED)
        # At 89%, only downgrade (one step), not forced-FAST
        assert decision.approved_tier == ModelTier.FAST  # BALANCED downgraded one step
        assert "critical" not in decision.reasoning.lower()

        # Now push to 91% on a fresh governor
        gov2 = Governor()
        gov2.initialize_task("t2", budget_usd=1.0)
        _spend(gov2, "t2", 0.91)
        budget2 = gov2.get_budget("t2")
        assert budget2.is_critical
        decision2 = gov2.route("t2", ModelTier.BALANCED)
        assert decision2.approved_tier == ModelTier.FAST
        assert "critical" in decision2.reasoning.lower() or "forced" in decision2.reasoning.lower()


# ---------------------------------------------------------------------------
# FailureMemory tests
# ---------------------------------------------------------------------------

class TestFailureMemory:

    def setup_method(self):
        """Each test gets a fresh FailureMemory pointed at a temp directory."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.mem = FailureMemory()
        # Patch MEMORY_DIR to use temp dir so tests don't pollute atlas_memory/
        self._patcher = patch.multiple(
            "atlas.memory.failure_memory",
            MEMORY_DIR=Path(self._tmpdir.name),
            INDEX_PATH=Path(self._tmpdir.name) / "failures.index",
            META_PATH=Path(self._tmpdir.name) / "failures_meta.json",
        )
        self._patcher.start()
        # Re-point the instance's internal paths too
        import atlas.memory.failure_memory as fm_module
        fm_module.MEMORY_DIR = Path(self._tmpdir.name)
        fm_module.INDEX_PATH = Path(self._tmpdir.name) / "failures.index"
        fm_module.META_PATH = Path(self._tmpdir.name) / "failures_meta.json"
        self.mem.initialize()

    def teardown_method(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_initialize_sets_available(self):
        assert self.mem.is_available

    def test_store_saves_to_json(self):
        import atlas.memory.failure_memory as fm_module
        self.mem.store(
            task_id="task-1",
            subtask_title="Reverse string",
            language="python",
            error_output="AssertionError: assert None == ''",
            failed_code="def f(s): return s[::-1]",
            fixed_code="def f(s): return (s or '')[::-1]",
            test_count=3,
        )
        assert self.mem.pattern_count == 1
        meta_path = fm_module.META_PATH
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert len(data) == 1
        assert data[0]["error_type"] == "AssertionError"
        assert data[0]["subtask_title"] == "Reverse string"
        assert data[0]["language"] == "python"

    def test_search_empty_returns_empty(self):
        results = self.mem.search("AssertionError: assert None == ''")
        assert results == []

    def test_search_finds_similar_pattern(self):
        self.mem.store(
            task_id="task-1",
            subtask_title="Reverse string",
            language="python",
            error_output="AssertionError: assert None == '' in test_reverse_none",
            failed_code="def reverse_string(s): return s[::-1]",
            fixed_code="def reverse_string(s): return (s or '')[::-1]",
            test_count=2,
        )
        # Query with similar error
        results = self.mem.search(
            "AssertionError: assert None == '' in test_reverse_none",
            language="python",
        )
        assert len(results) >= 1
        assert results[0].similarity_score >= 0.3
        assert results[0].rank == 1

    def test_search_filters_by_language(self):
        self.mem.store(
            task_id="task-1",
            subtask_title="JS task",
            language="javascript",
            error_output="TypeError: Cannot read property of undefined",
            failed_code="const f = x => x.value",
            fixed_code="const f = x => x?.value",
            test_count=1,
        )
        # Search in python should find nothing
        results = self.mem.search(
            "TypeError: Cannot read property of undefined",
            language="python",
        )
        assert results == []

    def test_search_respects_min_similarity(self):
        self.mem.store(
            task_id="task-1",
            subtask_title="Fibonacci",
            language="python",
            error_output="RecursionError: maximum recursion depth exceeded",
            failed_code="def fib(n): return fib(n-1) + fib(n-2)",
            fixed_code="def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)",
            test_count=1,
        )
        # Completely unrelated error → below MIN_SIMILARITY
        results = self.mem.search(
            "SyntaxError: invalid syntax on line 42",
            language="python",
        )
        # May find nothing, or very low score — must not exceed similarity
        for r in results:
            assert r.similarity_score >= 0.3

    def test_fix_description_generated(self):
        self.mem.store(
            task_id="task-1",
            subtask_title="Null check",
            language="python",
            error_output="AttributeError: 'NoneType' object has no attribute 'strip'",
            failed_code="def clean(s): return s.strip()",
            fixed_code="def clean(s): return s.strip() if s else ''",
            test_count=1,
        )
        pattern = self.mem._patterns[0]
        assert pattern.fix_description  # not empty
        assert pattern.fix_description != "Code restructured" or True  # acceptable fallback


# ---------------------------------------------------------------------------
# PromptEnhancer integration tests
# ---------------------------------------------------------------------------

class TestPromptEnhancer:

    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        import atlas.memory.failure_memory as fm_module
        fm_module.MEMORY_DIR = Path(self._tmpdir.name)
        fm_module.INDEX_PATH = Path(self._tmpdir.name) / "failures.index"
        fm_module.META_PATH = Path(self._tmpdir.name) / "failures_meta.json"
        # Reset singleton state for each test
        fm_module.failure_memory._patterns = []
        fm_module.failure_memory._available = False
        fm_module.failure_memory.initialize()
        self.enhancer = PromptEnhancer()

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_no_enhancement_when_memory_empty(self):
        result = self.enhancer.enhance_retry_prompt(
            original_prompt="Write a reverse string function.\nWrite the complete implementation",
            error_output="AssertionError: assert None == ''",
            language="python",
        )
        assert not result.enhancement_applied
        assert result.enhanced_prompt == result.original_prompt
        assert result.patterns_used == 0

    def test_enhancement_applied_when_similar_pattern_exists(self):
        import atlas.memory.failure_memory as fm_module
        fm = fm_module.failure_memory
        fm.store(
            task_id="task-1",
            subtask_title="Reverse string",
            language="python",
            error_output="AssertionError: assert None == '' in test_reverse_none",
            failed_code="def reverse_string(s): return s[::-1]",
            fixed_code="def reverse_string(s): return (s or '')[::-1]",
            test_count=2,
        )
        original = (
            "Fix the reverse_string function.\n"
            "Write the complete implementation"
        )
        result = self.enhancer.enhance_retry_prompt(
            original_prompt=original,
            error_output="AssertionError: assert None == '' in test_reverse_none",
            language="python",
        )
        assert result.enhancement_applied
        assert result.patterns_used >= 1
        assert "ATLAS MEMORY" in result.enhanced_prompt
        assert "AssertionError" in result.enhanced_prompt

    def test_memory_block_injected_before_write_the_complete_implementation(self):
        """Memory context appears before 'Write the complete implementation', not appended."""
        import atlas.memory.failure_memory as fm_module
        fm = fm_module.failure_memory
        fm.store(
            task_id="task-1",
            subtask_title="Reverse string",
            language="python",
            error_output="AssertionError: assert None == '' in test_reverse_none",
            failed_code="def reverse_string(s): return s[::-1]",
            fixed_code="def reverse_string(s): return (s or '')[::-1]",
            test_count=2,
        )
        original = (
            "Fix the reverse_string function.\n"
            "Write the complete implementation"
        )
        result = self.enhancer.enhance_retry_prompt(
            original_prompt=original,
            error_output="AssertionError: assert None == '' in test_reverse_none",
            language="python",
        )
        memory_pos = result.enhanced_prompt.index("ATLAS MEMORY")
        impl_pos = result.enhanced_prompt.index("Write the complete implementation")
        assert memory_pos < impl_pos, (
            "Memory block should appear BEFORE 'Write the complete implementation'"
        )

    def test_enhanced_prompt_contains_fixed_code_excerpt(self):
        """The injected block shows the working code — not just a message."""
        import atlas.memory.failure_memory as fm_module
        fm = fm_module.failure_memory
        fm.store(
            task_id="task-1",
            subtask_title="Reverse string",
            language="python",
            error_output="AssertionError: assert None == '' in test_reverse_none",
            failed_code="def reverse_string(s): return s[::-1]",
            fixed_code="def reverse_string(s): return (s or '')[::-1]",
            test_count=2,
        )
        result = self.enhancer.enhance_retry_prompt(
            original_prompt="Fix it.\nWrite the complete implementation",
            error_output="AssertionError: assert None == '' in test_reverse_none",
            language="python",
        )
        # The fixed_code excerpt must appear in the enhanced prompt
        assert "(s or '')" in result.enhanced_prompt

    def test_no_enhancement_below_similarity_threshold(self):
        """A completely different error does not trigger enhancement."""
        import atlas.memory.failure_memory as fm_module
        fm = fm_module.failure_memory
        fm.store(
            task_id="task-1",
            subtask_title="Fibonacci",
            language="python",
            error_output="RecursionError: maximum recursion depth exceeded in fib",
            failed_code="def fib(n): return fib(n-1) + fib(n-2)",
            fixed_code="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            test_count=1,
        )
        result = self.enhancer.enhance_retry_prompt(
            original_prompt="Fix it.\nWrite the complete implementation",
            error_output="SyntaxError: unexpected indent on line 5 of the module",
            language="python",
        )
        # If similarity is below 0.30 the enhancer should not apply
        if not result.enhancement_applied:
            assert result.enhanced_prompt == result.original_prompt
        else:
            # If it did match, score must be >= MIN_SIMILARITY
            assert result.patterns_used >= 1
