"""
atlas/governor/governor.py

Governor — cost-aware model routing and budget enforcement.

The Governor sits above all agents and makes two decisions:
  1. BEFORE a call: which model tier should this use given current budget?
  2. AFTER a call: record the spend, check if budget is at risk

Why this matters at scale:
  Without Governor: every subtask uses BALANCED regardless of task or budget.
  With Governor:
    - Simple subtasks get FAST (3x cheaper)
    - Hard subtasks get POWERFUL (better quality, worth the cost)
    - If we've spent 70% of budget by wave 1, downgrade wave 2 to FAST
    - If budget would be exceeded, abort the call before it happens

Budget tiers (approximate, based on Groq free tier):
  Groq free tier has rate limits, not dollar costs.
  We track token usage as a proxy for "computational budget."
  budget_usd in TaskRequest maps to a token allowance.

Real-time tracking:
  Every LLM call updates the BudgetState.
  The Governor checks BudgetState before approving each call.
  The dashboard shows live spend via WebSocket cost.update events.

Design:
  Governor is a singleton — one instance tracks ALL active tasks.
  BudgetState is per task_id — isolated accounting per task.
"""

from __future__ import annotations

from collections import defaultdict

from atlas.config import settings
from atlas.contracts import ModelTier
from atlas.contracts_v3 import BudgetState, GovernorDecision
from atlas.logging import get_logger

log = get_logger(__name__)

# Token allowance per USD budget (approximate for Groq free tier)
# Since Groq is free, we use token count as the budget proxy
TOKENS_PER_USD = 500_000  # 500k tokens per $1 of declared budget

# Budget threshold for tier downgrade decisions
DOWNGRADE_THRESHOLD = 0.70    # Downgrade one step at 70% budget used
# Critical threshold (force FAST) is defined on BudgetState.is_critical (90%)


class Governor:
    """
    Manages model routing and budget enforcement across all tasks.

    One singleton for the whole application.
    Per-task state stored in self._budgets.
    """

    def __init__(self) -> None:
        # task_id → BudgetState
        self._budgets: dict[str, BudgetState] = {}

    def initialize_task(self, task_id: str, budget_usd: float) -> BudgetState:
        """
        Set up budget tracking for a new task.

        Called at pipeline start before any LLM calls.
        """
        state = BudgetState(task_id=task_id, budget_usd=budget_usd)
        self._budgets[task_id] = state
        log.info(
            "governor_task_initialized",
            task_id=task_id,
            budget_usd=budget_usd,
            token_allowance=int(budget_usd * TOKENS_PER_USD),
        )
        return state

    def route(
        self,
        task_id: str,
        requested_tier: ModelTier,
        context: str = "",
    ) -> GovernorDecision:
        """
        Approve or downgrade a model tier request.

        Called before every LLM call in CoderAgent and ArchitectAgent.
        Returns a GovernorDecision with the approved model.

        Args:
            task_id: The task this call is for
            requested_tier: What the agent wants
            context: Optional description for logging ("subtask st_1 attempt 2")

        Returns:
            GovernorDecision — always returns a valid model to use.
        """
        budget = self._get_or_create_budget(task_id)
        percent_used = budget.percent_used

        # Determine approved tier based on budget state
        approved_tier = requested_tier
        downgraded = False
        reasoning = "Budget healthy — approved as requested"

        if budget.is_exhausted:
            # Hard stop — shouldn't happen, but safety net
            approved_tier = ModelTier.FAST
            downgraded = requested_tier != ModelTier.FAST
            reasoning = f"Budget exhausted ({percent_used:.0f}% used) — forced to FAST"

        elif budget.is_critical:
            # Critical (>=90%) — force FAST for everything
            approved_tier = ModelTier.FAST
            downgraded = requested_tier != ModelTier.FAST
            reasoning = f"Critical budget ({percent_used:.0f}% used) — forced to FAST"

        elif percent_used >= DOWNGRADE_THRESHOLD * 100:
            # Warning — downgrade one tier
            if requested_tier == ModelTier.POWERFUL:
                approved_tier = ModelTier.BALANCED
                downgraded = True
                reasoning = f"Budget at {percent_used:.0f}% — downgraded POWERFUL → BALANCED"
            elif requested_tier == ModelTier.BALANCED:
                approved_tier = ModelTier.FAST
                downgraded = True
                reasoning = f"Budget at {percent_used:.0f}% — downgraded BALANCED → FAST"

        model = self._model_for_tier(approved_tier)

        if downgraded:
            budget_copy = self._budgets.get(task_id, budget)
            updated = budget_copy.model_copy(update={"tier_downgrades": budget_copy.tier_downgrades + 1})
            self._budgets[task_id] = updated

        decision = GovernorDecision(
            requested_tier=requested_tier,
            approved_tier=approved_tier,
            model=model,
            reasoning=reasoning,
            budget_state=budget,
            downgraded=downgraded,
        )

        if downgraded:
            log.warning(
                "governor_tier_downgraded",
                task_id=task_id,
                requested=requested_tier,
                approved=approved_tier,
                percent_used=round(percent_used, 1),
                context=context,
            )
        else:
            log.debug(
                "governor_tier_approved",
                task_id=task_id,
                tier=approved_tier,
                percent_used=round(percent_used, 1),
            )

        return decision

    def record_usage(
        self,
        task_id: str,
        tokens_in: int,
        tokens_out: int,
        model: str,
    ) -> BudgetState:
        """
        Record actual token usage after an LLM call completes.

        Called by BaseAgent after every successful completion.
        Updates the running cost total and emits a cost event.
        """
        budget = self._get_or_create_budget(task_id)

        # Convert tokens to approximate USD spend
        cost_usd = self._estimate_cost(model, tokens_in, tokens_out)

        updated = budget.model_copy(update={
            "spent_usd": budget.spent_usd + cost_usd,
            "llm_calls": budget.llm_calls + 1,
            "tokens_in": budget.tokens_in + tokens_in,
            "tokens_out": budget.tokens_out + tokens_out,
        })
        self._budgets[task_id] = updated

        log.debug(
            "governor_usage_recorded",
            task_id=task_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost_usd, 6),
            total_spent=round(updated.spent_usd, 6),
            percent_used=round(updated.percent_used, 1),
        )

        return updated

    def get_budget(self, task_id: str) -> BudgetState | None:
        """Get current budget state for a task."""
        return self._budgets.get(task_id)

    def get_all_budgets(self) -> dict[str, BudgetState]:
        """All active task budgets — used by the dashboard."""
        return dict(self._budgets)

    def _get_or_create_budget(self, task_id: str) -> BudgetState:
        """Get existing budget or create a default one."""
        if task_id not in self._budgets:
            # Task started before Governor was initialized — use default
            self._budgets[task_id] = BudgetState(
                task_id=task_id,
                budget_usd=0.50,  # Default
            )
        return self._budgets[task_id]

    @staticmethod
    def _model_for_tier(tier: ModelTier) -> str:
        return {
            ModelTier.FAST:     settings.model_fast,
            ModelTier.BALANCED: settings.model_balanced,
            ModelTier.POWERFUL: settings.model_powerful,
        }[tier]

    @staticmethod
    def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
        """
        Approximate cost in USD.
        Groq is free but we track tokens as budget proxy.
        This enables the same code to work with paid providers.
        """
        pricing: dict[str, tuple[float, float]] = {
            "llama-3.1-8b-instant":          (0.05, 0.08),
            "llama-3.3-70b-versatile":       (0.59, 0.79),
            "deepseek-r1-distill-llama-70b": (0.75, 0.99),
        }
        in_price, out_price = pricing.get(model, (0.59, 0.79))
        return (tokens_in * in_price + tokens_out * out_price) / 1_000_000


# Singleton
governor = Governor()
