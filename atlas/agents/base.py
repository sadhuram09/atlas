"""
atlas/agents/base.py

BaseAgent — the abstract class every ATLAS agent inherits from.

LLM provider: Groq (free tier)
  Groq exposes an OpenAI-compatible API, so we use the `openai` Python SDK
  pointed at Groq's base URL. Zero code changes needed if you ever want to
  switch to actual OpenAI or another compatible provider — just change the
  base_url and api_key in config.py.

Architecture contract:
  Every agent MUST implement run(). Everything else is inherited:
  - Structured logging with automatic task_id context
  - Groq LLM client (OpenAI-compatible, free)
  - Token tracking (Groq returns usage just like OpenAI)
  - Retry logic via tenacity (handles rate limits gracefully)
  - Consistent error handling

Why abstract?
  The Governor, Orchestrator, Architect, Coder, and Verifier all share
  this base. Changing logging or retry logic here changes it everywhere.
  Agents never call each other directly — they go through the Orchestrator.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from atlas.config import settings
from atlas.contracts import AgentMessage, AgentRole, CostEvent, ModelTier
from atlas.logging import get_logger


class AgentError(Exception):
    """Raised when an agent hits an unrecoverable error."""
    def __init__(self, message: str, agent: AgentRole, task_id: str) -> None:
        self.agent = agent
        self.task_id = task_id
        super().__init__(f"[{agent}][{task_id}] {message}")


class BaseAgent(ABC):
    """
    Abstract base for all ATLAS agents.

    Subclass and implement `run()`. Everything else is handled here.

    Example:
        class CoderAgent(BaseAgent):
            role = AgentRole.CODER

            async def run(self, message: AgentMessage) -> AgentMessage:
                code = await self.complete("Write a Python function that...")
                return self._reply(message, {"code": code})
    """

    role: AgentRole  # Must be set on the subclass

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.log: structlog.stdlib.BoundLogger = get_logger(
            f"atlas.agent.{self.role}"
        ).bind(task_id=task_id, agent=self.role)

        # Groq client via OpenAI SDK — lazy, one per agent instance
        self._groq: AsyncOpenAI | None = None

        # Cost/usage accumulator — reset per task
        # Groq is free but we track tokens for the dashboard anyway
        self._total_tokens: int = 0
        self._cost_events: list[CostEvent] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and return a reply.

        This is the only method subclasses must implement.
        All other helpers are available via `self`.
        """
        ...

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    @property
    def groq_client(self) -> AsyncOpenAI:
        """
        Lazy Groq client — created once per agent instance.

        We use the openai SDK with Groq's base_url.
        This is the officially recommended way to call Groq from Python.
        """
        if self._groq is None:
            self._groq = AsyncOpenAI(
                api_key=settings.groq_key,
                base_url=settings.groq_base_url,
            )
        return self._groq

    def _model_for_tier(self, tier: ModelTier) -> str:
        """Map a cost tier to the actual Groq model string."""
        return {
            ModelTier.FAST:     settings.model_fast,
            ModelTier.BALANCED: settings.model_balanced,
            ModelTier.POWERFUL: settings.model_powerful,
        }[tier]

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        tier: ModelTier = ModelTier.BALANCED,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        """
        Call Groq and return the text response.

        Automatically:
          - Selects model based on `tier` (fast/balanced/powerful)
          - Retries on rate limits (Groq free tier: 30 req/min)
          - Tracks token usage for the dashboard
          - Logs every call with timing

        Args:
            prompt: The user-turn content.
            system: Optional system prompt.
            tier: ModelTier.FAST / BALANCED / POWERFUL.
            max_tokens: Upper bound on response length.
            temperature: 0.0 = deterministic, 1.0 = creative.

        Returns:
            The model's text response as a string.

        Groq rate limits (free tier):
            llama-3.1-8b-instant:       30 req/min, 131k tokens/min
            llama-3.3-70b-versatile:    30 req/min,  6k tokens/min
            deepseek-r1-distill-70b:    30 req/min,  6k tokens/min
        """
        model = self._model_for_tier(tier)
        start = time.monotonic()

        self.log.info(
            "llm_call_start",
            model=model,
            prompt_chars=len(prompt),
            tier=tier,
        )

        messages = [
            {
                "role": "system",
                "content": system or "You are ATLAS, a precise code-writing AI agent.",
            },
            {"role": "user", "content": prompt},
        ]

        response = None

        # Retry on transient errors with exponential backoff (B5).
        # Groq free tier: 30 req/min — connection/timeout errors are included
        # so a flaky network doesn't immediately fail the task.
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(
                    (RateLimitError, APIStatusError, APIConnectionError, APITimeoutError)
                ),
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=1, min=3, max=60),
                reraise=True,
            ):
                with attempt:
                    response = await self.groq_client.chat.completions.create(
                        model=model,
                        messages=messages,      # type: ignore[arg-type]
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
        except RateLimitError:
            raise AgentError(
                "Groq rate limit reached after retries — no budget consumed; wait 60s then retry",
                self.role, self.task_id,
            ) from None
        except (APIConnectionError, APITimeoutError) as exc:
            raise AgentError(
                f"Groq API unreachable ({type(exc).__name__}) — no budget consumed; safe to retry",
                self.role, self.task_id,
            ) from exc
        except APIStatusError as exc:
            raise AgentError(
                f"Groq API error (HTTP {exc.status_code}) after retries — partial budget may have been consumed",
                self.role, self.task_id,
            ) from exc

        duration = (time.monotonic() - start) * 1000
        usage = response.usage  # type: ignore[union-attr]

        tokens_in  = usage.prompt_tokens     if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        self._total_tokens += tokens_in + tokens_out

        # Groq is free — cost is $0.00, but we record it for dashboard
        # consistency (and future-proofing if you ever switch providers).
        cost_event = CostEvent(
            task_id=self.task_id,
            agent=self.role,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,  # Groq free tier
        )
        self._cost_events.append(cost_event)

        self.log.info(
            "llm_call_complete",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=round(duration),
            provider="groq",
        )

        return response.choices[0].message.content or ""  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _reply(
        self,
        incoming: AgentMessage,
        payload: dict[str, Any],
        to: AgentRole | None = None,
    ) -> AgentMessage:
        """
        Construct a typed reply message.

        Automatically flips from_agent / to_agent.
        Pass `to` explicitly when routing to a non-sender.
        """
        return AgentMessage(
            task_id=self.task_id,
            from_agent=self.role,
            to_agent=to or incoming.from_agent,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BaseAgent":
        self.log.info("agent_started")
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self.log.error(
                "agent_failed",
                error=str(exc),
                error_type=exc_type.__name__,
                total_tokens=self._total_tokens,
            )
        else:
            self.log.info(
                "agent_completed",
                total_tokens=self._total_tokens,
                cost_events=len(self._cost_events),
                provider="groq",
            )

