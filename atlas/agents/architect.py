"""
atlas/agents/architect.py

ArchitectAgent — decomposes a complex task into a dependency DAG.

This is fundamentally different from a simple task list.

Example — "Build a URL shortener":
  Flat list approach (naive):
    1. Write the shortener function
    2. Write the database layer
    3. Write the API endpoints
    4. Write the tests
    → All sequential. Slow.

  DAG approach (ATLAS):
    Node A: Write URL shortener core logic    (no deps)
    Node B: Write database schema             (no deps)
    Node C: Write API endpoints               (deps: A, B)
    Node D: Write integration tests           (deps: C)

    Wave 0: A + B run in PARALLEL
    Wave 1: C runs (both A and B complete)
    Wave 2: D runs (C complete)

  Result: A and B execute simultaneously. Potentially 2x faster.
  For 4+ independent subtasks, the speedup compounds.

The ArchitectAgent also assigns a ModelTier to each subtask individually.
A subtask that's "write a simple helper function" gets FAST.
A subtask that's "implement the core algorithm" gets BALANCED or POWERFUL.
"""

from __future__ import annotations

import json
import re

from atlas.agents.base import BaseAgent
from atlas.contracts import AgentMessage, AgentRole, ModelTier
from atlas.contracts_v2 import ComplexityScore, ExecutionPlan, SubTaskNode, SubTaskStatus
from atlas.logging import get_logger

log = get_logger(__name__)

ARCHITECT_SYSTEM_PROMPT = """You are ATLAS ArchitectAgent, a software systems designer.

Your job: decompose a coding task into a dependency graph of subtasks.

Respond with ONLY valid JSON in this exact format:
{
  "subtasks": [
    {
      "id": "st_1",
      "title": "Short title",
      "description": "Detailed description of exactly what to implement",
      "acceptance_criteria": ["criterion 1", "criterion 2"],
      "depends_on": [],
      "order": 0,
      "tier": "fast|balanced|powerful"
    }
  ]
}

RULES:
- Use exactly the IDs "st_1", "st_2", etc.
- depends_on contains IDs of subtasks that must complete first
- No circular dependencies
- Each subtask must be implementable independently given its deps
- acceptance_criteria: specific, testable requirements (not vague)
- tier: fast for simple helpers, balanced for logic, powerful for algorithms
- For simple tasks (score 1-3): return exactly 1 subtask with depends_on: []
- For moderate tasks (score 4-6): 2-3 subtasks
- For complex tasks (score 7-10): 3-6 subtasks with real dependencies
"""


class ArchitectAgent(BaseAgent):
    """
    Decomposes a task into a dependency DAG.

    The DAG drives parallel execution in the Orchestrator.
    Subtasks with no blocking dependencies run simultaneously.
    """

    role = AgentRole.ARCHITECT

    async def run(self, message: AgentMessage) -> AgentMessage:
        """
        Produce an ExecutionPlan for a task.

        Payload in:
            task_title: str
            task_description: str
            language: str
            complexity: dict  (ComplexityScore from IntelligenceRouter)

        Payload out:
            plan: dict  (serialised ExecutionPlan)
        """
        payload = message.payload
        title = payload["task_title"]
        description = payload["task_description"]
        language = payload.get("language", "python")
        complexity = ComplexityScore(**payload["complexity"])

        self.log.info(
            "architect_started",
            title=title,
            complexity_score=complexity.score,
            estimated_subtasks=complexity.estimated_subtasks,
        )

        prompt = f"""Task: {title}
Language: {language}
Complexity score: {complexity.score}/10
Estimated subtasks: {complexity.estimated_subtasks}

Description:
{description}

Decompose this into {complexity.estimated_subtasks} subtasks with dependencies.
Return JSON only."""

        raw = await self.complete(
            prompt=prompt,
            system=ARCHITECT_SYSTEM_PROMPT,
            tier=ModelTier.BALANCED,
            max_tokens=2048,
            temperature=0.1,
        )

        nodes = self._parse_subtasks(raw, complexity)
        plan = self._build_plan(
            task_id=self.task_id,
            nodes=nodes,
            complexity=complexity,
        )

        self.log.info(
            "architect_completed",
            subtasks=len(plan.nodes),
            waves=len(plan.execution_waves),
            edges=len(plan.edges),
        )

        return self._reply(
            message,
            payload={"plan": plan.model_dump(mode="json")},
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_subtasks(
        self, raw: str, complexity: ComplexityScore
    ) -> list[SubTaskNode]:
        """Parse LLM JSON response into SubTaskNode objects."""
        clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")

        try:
            data = json.loads(clean)
            subtasks_raw = data.get("subtasks", [])
        except (json.JSONDecodeError, KeyError):
            self.log.warning("architect_parse_failed", raw=raw[:300])
            # Fallback: single subtask covering the whole task
            return self._single_subtask_fallback(complexity)

        if not subtasks_raw:
            return self._single_subtask_fallback(complexity)

        tier_map = {
            "fast": ModelTier.FAST,
            "balanced": ModelTier.BALANCED,
            "powerful": ModelTier.POWERFUL,
        }

        nodes: list[SubTaskNode] = []
        for i, s in enumerate(subtasks_raw):
            tier = tier_map.get(str(s.get("tier", "balanced")).lower(), ModelTier.BALANCED)

            node = SubTaskNode(
                id=str(s.get("id", f"st_{i+1}")),
                title=str(s.get("title", f"Subtask {i+1}")),
                description=str(s.get("description", "")),
                acceptance_criteria=[str(c) for c in s.get("acceptance_criteria", [])],
                depends_on=[str(d) for d in s.get("depends_on", [])],
                order=int(s.get("order", i)),
                status=SubTaskStatus.WAITING,
                assigned_tier=tier,
            )
            nodes.append(node)

        return nodes

    def _single_subtask_fallback(self, complexity: ComplexityScore) -> list[SubTaskNode]:
        """Return a single all-encompassing subtask — safe fallback."""
        return [
            SubTaskNode(
                id="st_1",
                title="Complete implementation",
                description="Implement the full solution as described",
                depends_on=[],
                order=0,
                status=SubTaskStatus.READY,
                assigned_tier=complexity.tier,
            )
        ]

    # ------------------------------------------------------------------
    # DAG construction
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        task_id: str,
        nodes: list[SubTaskNode],
        complexity: ComplexityScore,
    ) -> ExecutionPlan:
        """
        Build the ExecutionPlan with dependency edges and execution waves.

        Wave computation (topological sort):
          Wave 0: nodes with empty depends_on
          Wave 1: nodes whose all deps are in wave 0
          Wave N: nodes whose all deps are in waves 0..N-1

        Nodes in the same wave have no dependencies on each other
        and can execute in parallel safely.
        """
        # Build edges list
        edges: list[tuple[str, str]] = []
        for node in nodes:
            for dep_id in node.depends_on:
                edges.append((dep_id, node.id))

        # Compute execution waves via topological sort (Kahn's algorithm)
        waves = self._compute_waves(nodes)

        # Mark wave-0 nodes as READY (all others stay WAITING)
        node_map = {n.id: n for n in nodes}
        updated_nodes: list[SubTaskNode] = []

        for node in nodes:
            if not node.depends_on:
                # No dependencies → immediately ready
                updated = node.model_copy(update={"status": SubTaskStatus.READY})
            else:
                updated = node
            updated_nodes.append(updated)

        return ExecutionPlan(
            task_id=task_id,
            nodes=updated_nodes,
            edges=edges,
            execution_waves=waves,
            complexity=complexity,
        )

    @staticmethod
    def _compute_waves(nodes: list[SubTaskNode]) -> list[list[str]]:
        """
        Compute execution waves using Kahn's topological sort.

        Returns a list of waves, where each wave is a list of node IDs
        that can execute in parallel.
        """
        # Build in-degree map
        all_ids = {n.id for n in nodes}
        in_degree: dict[str, int] = {n.id: 0 for n in nodes}
        adjacency: dict[str, list[str]] = {n.id: [] for n in nodes}

        for node in nodes:
            for dep_id in node.depends_on:
                if dep_id in all_ids:
                    in_degree[node.id] += 1
                    adjacency[dep_id].append(node.id)

        waves: list[list[str]] = []
        remaining = set(all_ids)

        while remaining:
            # Current wave: all remaining nodes with in_degree 0
            wave = [nid for nid in remaining if in_degree[nid] == 0]

            if not wave:
                # Cycle detected — add remaining as final wave
                waves.append(list(remaining))
                break

            waves.append(sorted(wave))  # Sort for determinism

            # Remove this wave and update in_degrees
            for nid in wave:
                remaining.discard(nid)
                for neighbor in adjacency[nid]:
                    in_degree[neighbor] -= 1

        return waves
