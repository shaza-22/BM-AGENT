"""
LLM access for the navigation agent.

What it does
    Defines the narrow interface the agent uses to talk to a language model,
    plus two implementations: a Claude-backed client and a scripted fake.

Inputs
    ``complete(prompt, system=..., schema=...)`` -- a rendered prompt, an
    optional system prompt, and an optional JSON Schema the reply must satisfy.

Outputs
    The model's reply as a string. Callers parse it; this layer does not.

Why it is needed
    Two reasons. Swappability: selection logic must not know which model it is
    talking to, so switching models or providers touches only this file.
    Testability: the entire navigation suite runs offline against
    :class:`FakeLLMClient`, with no API key and no network.

Note on the optional dependency
    ``anthropic`` is imported lazily inside the call, matching how Playwright
    is handled in the browsing layer: the project imports, its tests run, and
    the fixtures scripts work on a machine that has never installed the SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, Sequence

from agent import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or refuses to answer.

    The navigator catches this and ends the run with ``status="error"`` rather
    than letting an API problem crash a task mid-way.
    """


class LLMClient(Protocol):
    """The only thing the agent needs from a language model."""

    def complete(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> str:
        """Return the model's reply as text."""


class ClaudeLLMClient:
    """Claude-backed :class:`LLMClient`.

    ``schema`` is passed through as a structured-output constraint, so the
    reply is valid JSON matching it. The selector still parses defensively --
    a different ``LLMClient`` implementation may offer no such guarantee.
    """

    def __init__(
        self,
        *,
        model: str = config.MODEL,
        effort: str = config.EFFORT,
        max_tokens: int = config.MAX_TOKENS,
        client: Any | None = None,
        enable_fallbacks: bool = True,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client
        # Server-side refusal fallback: if a safety classifier declines, the
        # same request is re-run on another model inside the same call instead
        # of the navigation step simply failing.
        self._enable_fallbacks = enable_fallbacks
        self.calls = 0

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError(
                    "the 'anthropic' package is not installed -- "
                    "run `pip install anthropic`, or pass a different LLMClient"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> str:
        client = self._ensure_client()

        output_config: dict[str, Any] = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            request["system"] = system

        self.calls += 1
        response = self._send(client, request)

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMError(f"model refused the request (category={getattr(details, 'category', None)})")

        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    def _send(self, client: Any, request: dict[str, Any]) -> Any:
        """Send the request, degrading gracefully if fallbacks are unsupported."""
        if self._enable_fallbacks:
            try:
                return client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **request,
                )
            except Exception as exc:
                if not _looks_like_unsupported_parameter(exc):
                    raise LLMError(f"Claude request failed: {exc}") from exc
                # An older SDK or endpoint that does not know the parameter --
                # disable it for the rest of the run rather than failing.
                logger.warning("refusal fallbacks unsupported here (%s); continuing without", exc)
                self._enable_fallbacks = False

        try:
            return client.messages.create(**request)
        except Exception as exc:
            raise LLMError(f"Claude request failed: {exc}") from exc


def _looks_like_unsupported_parameter(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, TypeError) or "beta" in message or "fallback" in message


class FakeLLMClient:
    """Scripted :class:`LLMClient` for tests and offline demos.

    Accepts either a sequence of replies (returned in order, the last one
    repeating once exhausted) or a callable receiving the prompt.
    """

    def __init__(self, responses: Sequence[str] | Callable[[str], str]) -> None:
        self._responses = responses
        self._index = 0
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.schemas: list[dict | None] = []

    def complete(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.schemas.append(schema)

        if callable(self._responses):
            return self._responses(prompt)
        if not self._responses:
            raise LLMError("FakeLLMClient was given no responses")
        index = min(self._index, len(self._responses) - 1)
        self._index += 1
        return self._responses[index]

    @property
    def calls(self) -> int:
        return len(self.prompts)
