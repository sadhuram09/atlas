"""
atlas/agents/intelligence_router.py

IntelligenceRouter — the first thing that runs on every task.

Before a single line of code is written, the router:
  1. Analyses the task description
  2. Scores complexity 1-10
  3. Decides how many subtasks to decompose it into
  4. Picks the right model tier for each subtask
  5. Decides if parallel execution is worth it

Why this matters:
  Without routing, every task uses the same model.
  With routing:
    - "Write an add() function" → score 1 → llama-3.1-8b (fast, free)
    - "Build a REST API with auth" → score 7 → llama-3.3-70b (balanced)
    - "Implement a B-tree from scratch" → score 9 → deepseek-r1 (reasoning)

  This is what makes ATLAS economically viable at scale.
  Hard tasks get more compute. Simple tasks don't waste it.

The router itself uses the FAST model — scoring is cheap.
It's analysing text, not writing code.
"""

from __future__ import annotations

import json
import re

from atlas.agents.base import BaseAgent
from atlas.contracts import AgentMessage, AgentRole, ModelTier
from atlas.contracts_v2 import ComplexityScore
from atlas.logging import get_logger

log = get_logger(__name__)

ROUTER_SYSTEM_PROMPT = """You are ATLAS IntelligenceRouter, a task complexity analyser.

Your job: analyse a coding task and score its complexity.

Respond with ONLY valid JSON in this exact format — no explanation, no markdown:
{
  "score": <integer 1-10>,
  "tier": "<fast|balanced|powerful>",
  "reasoning": "<one sentence explaining the score>",
  "estimated_subtasks": <integer 1-8>,
  "requires_parallel": <true|false>
}

Scoring guide:
  1-3 (fast):     Single function, trivial logic, no edge cases
  4-6 (balanced): Multiple functions, moderate logic, some edge cases  
  7-8 (balanced): Complex algorithms, multiple interacting components
  9-10 (powerful): Advanced algorithms, data structures, systems design

requires_parallel: true if estimated_subtasks >= 3 and subtasks are independent
"""


class IntelligenceRouter(BaseAgent):
    """
    Scores task complexity and decides model routing before work begins.

    Uses the FAST model (llama-3.1-8b-instant) — scoring is cheap.
    Takes ~0.5 seconds. Saves potentially many seconds on the actual work
    by ensuring the right model is used.
    """

    role = AgentRole.GOVERNOR

    async def run(self, message: AgentMessage) -> AgentMessage:
        """
        Score a task and return a ComplexityScore.

        Payload in:
            task_title: str
            task_description: str
            language: str

        Payload out:
            complexity: dict  (serialised ComplexityScore)
        """
        payload = message.payload
        title = payload["task_title"]
        description = payload["task_description"]
        language = payload.get("language", "python")

        self.log.info("router_scoring", title=title)

        prompt = f"""Task: {title}
Language: {language}
Description: {description}

Score this task's complexity and return JSON only."""

        raw = await self.complete(
            prompt=prompt,
            system=ROUTER_SYSTEM_PROMPT,
            tier=ModelTier.FAST,  # Scoring is cheap
            max_tokens=256,
            temperature=0.0,  # Fully deterministic scoring
        )

        complexity = self._parse_complexity(raw, title)

        self.log.info(
            "router_scored",
            score=complexity.score,
            tier=complexity.tier,
            estimated_subtasks=complexity.estimated_subtasks,
            requires_parallel=complexity.requires_parallel,
        )

        return self._reply(
            message,
            payload={"complexity": complexity.model_dump()},
        )

    def _parse_complexity(self, raw: str, title: str) -> ComplexityScore:
        """Parse the LLM JSON response into a ComplexityScore."""
        # Strip markdown fences if the model added them despite instructions
        clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")

        try:
            data = json.loads(clean)
            score = max(1, min(10, int(data.get("score", 5))))
            tier_str = data.get("tier", "balanced").lower()

            tier_map = {
                "fast": ModelTier.FAST,
                "balanced": ModelTier.BALANCED,
                "powerful": ModelTier.POWERFUL,
            }
            tier = tier_map.get(tier_str, ModelTier.BALANCED)

            return ComplexityScore(
                score=score,
                tier=tier,
                reasoning=data.get("reasoning", ""),
                estimated_subtasks=max(1, min(8, int(data.get("estimated_subtasks", 1)))),
                requires_parallel=bool(data.get("requires_parallel", False)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.log.warning(
                "router_parse_failed",
                error=str(e),
                raw=raw[:200],
            )
            # Safe default — assume moderate complexity
            return ComplexityScore(
                score=5,
                tier=ModelTier.BALANCED,
                reasoning="Could not parse complexity score — defaulting to balanced",
                estimated_subtasks=1,
                requires_parallel=False,
            )
