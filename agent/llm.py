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
import os
import pathlib
import re
from typing import Any, Callable, Protocol, Sequence

from agent import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or refuses to answer.

    The navigator catches this and ends the run with ``status="error"`` rather
    than letting an API problem crash a task mid-way.
    """


def load_project_env() -> None:
    """Load ``.env`` from the project root, if python-dotenv is installed.

    Existing environment variables win, so an exported key is never overridden
    by the file. A missing python-dotenv is not an error: the key may already
    be in the environment.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; relying on the ambient environment")
        return
    load_dotenv(PROJECT_ROOT / ".env")


def resolve_api_key(env_var: str, provider: str) -> str:
    """Fetch an API key from the environment, or explain how to supply one.

    The return value is handed straight to the SDK and never stored on a client
    instance, logged, or included in an error message. Only the *name* of the
    variable appears in the guidance below.
    """
    load_project_env()
    key = (os.environ.get(env_var) or "").strip()
    if not key:
        raise LLMError(
            f"no {provider} API key found: set {env_var} in the environment, or put "
            f"{env_var}=<your key> in {PROJECT_ROOT / '.env'} (that file is gitignored)"
        )
    return key


def scrub(text: str) -> str:
    """Remove anything that looks like an API key from a message.

    Belt and braces: no key is ever passed to a log or an exception on purpose,
    but SDK errors sometimes echo request details back, and an error message is
    exactly the kind of text that ends up pasted into an issue.
    """
    text = _GOOGLE_KEY_RE.sub("<redacted>", text)
    text = _ANTHROPIC_KEY_RE.sub("<redacted>", text)
    return _KEY_QUERY_RE.sub(r"\1<redacted>", text)


_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")
_ANTHROPIC_KEY_RE = re.compile(r"sk-ant-[0-9A-Za-z_\-]{10,}")
_KEY_QUERY_RE = re.compile(r"([?&](?:key|api_key|access_token)=)[^&\s\"']+", re.IGNORECASE)


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
        model: str = config.CLAUDE_MODEL,
        effort: str = config.CLAUDE_EFFORT,
        max_tokens: int = config.CLAUDE_MAX_TOKENS,
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
            except ImportError:
                raise LLMError(
                    "the 'anthropic' package is not installed -- "
                    "run `pip install anthropic`, or pass a different LLMClient"
                ) from None
            self._client = anthropic.Anthropic(
                api_key=resolve_api_key(config.ANTHROPIC_API_KEY_ENV, "Anthropic")
            )
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

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            # An empty reply is an API problem, not a decision. Returning it
            # would parse as "no candidate fits", which reads as "the site does
            # not cover this" -- a materially different and wrong conclusion.
            raise LLMError(
                f"Claude returned no text (stop_reason={getattr(response, 'stop_reason', None)})"
            )
        return text

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
                    raise LLMError(f"Claude request failed: {scrub(str(exc))}") from None
                # An older SDK or endpoint that does not know the parameter --
                # disable it for the rest of the run rather than failing.
                logger.warning("refusal fallbacks unsupported here (%s); continuing without", exc)
                self._enable_fallbacks = False

        try:
            return client.messages.create(**request)
        except Exception as exc:
            # `from None` on purpose: chaining would put the raw SDK error into
            # every traceback, unscrubbed.
            raise LLMError(f"Claude request failed: {scrub(str(exc))}") from None


def _looks_like_unsupported_parameter(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, TypeError) or "beta" in message or "fallback" in message


class _SchemaRejected(Exception):
    """Internal: the API would not accept the structured-output schema."""


class GeminiLLMClient:
    """Gemini-backed :class:`LLMClient`.

    Structured output goes through ``response_json_schema``, which takes the
    same JSON Schema dict the Claude client uses, so both providers can share
    one schema. Gemini is a different model on a different API, though, so the
    reply may still be shaped differently -- ``link_selector.parse_selection``
    is the safety net for both, and is tested against each client.

    The API key is read from the environment (or ``.env``) at connect time and
    handed straight to the SDK. It is never stored on this object, so it cannot
    reach a log line, a repr, or an error message.
    """

    def __init__(
        self,
        *,
        model: str = config.GEMINI_MODEL,
        max_tokens: int = config.GEMINI_MAX_TOKENS,
        client: Any | None = None,
        api_key_env: str = config.GEMINI_API_KEY_ENV,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self._api_key_env = api_key_env
        # Cleared if the API turns out not to accept the schema, so the run
        # continues on the defensive parser instead of failing every hop.
        self._structured_output = True
        self.calls = 0

    def __repr__(self) -> str:
        # Explicit, so no future attribute can leak into a log through repr().
        return f"GeminiLLMClient(model={self.model!r})"

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise LLMError(
                    "the 'google-genai' package is not installed -- "
                    "run `pip install google-genai`, or pass a different LLMClient"
                ) from None
            self._client = genai.Client(
                api_key=resolve_api_key(self._api_key_env, "Gemini")
            )
        return self._client

    def complete(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> str:
        client = self._ensure_client()
        self.calls += 1

        if schema is not None and self._structured_output:
            try:
                return self._generate(client, prompt, system, schema)
            except _SchemaRejected as exc:
                logger.warning(
                    "Gemini did not accept the response schema (%s); "
                    "continuing without it and relying on defensive parsing",
                    exc,
                )
                self._structured_output = False

        return self._generate(client, prompt, system, None)

    def _generate(
        self, client: Any, prompt: str, system: str | None, schema: dict | None
    ) -> str:
        # Passed as a plain dict rather than types.GenerateContentConfig: the
        # SDK accepts either, and this keeps the whole call path free of any
        # google.genai import, so an injected client works on a machine that
        # does not have the package at all.
        settings: dict[str, Any] = {"max_output_tokens": self.max_tokens}
        if system:
            settings["system_instruction"] = system
        if schema is not None:
            settings["response_mime_type"] = "application/json"
            settings["response_json_schema"] = schema

        try:
            response = client.models.generate_content(
                model=self.model, contents=prompt, config=settings
            )
        except Exception as exc:
            message = scrub(str(exc))
            if schema is not None and _looks_like_schema_rejection(message):
                raise _SchemaRejected(message) from None
            # `from None` on purpose: chaining would put the raw SDK error into
            # every traceback, unscrubbed.
            raise LLMError(f"Gemini request failed: {message}") from None

        text = response.text
        if not text:
            raise LLMError(f"Gemini returned no text ({_finish_reason(response)})")
        return text


def _looks_like_schema_rejection(message: str) -> bool:
    lowered = message.lower()
    return "schema" in lowered or "response_mime_type" in lowered


def _finish_reason(response: Any) -> str:
    """Why an empty reply came back -- a safety block, a token cap, or unknown."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason:
            return f"finish_reason={reason}"
    feedback = getattr(response, "prompt_feedback", None)
    blocked = getattr(feedback, "block_reason", None)
    if blocked:
        return f"blocked={blocked}"
    return "no finish reason reported"


def make_llm_client(provider: str | None = None, **overrides: Any) -> LLMClient:
    """Build the configured provider's client.

    Selecting the provider here rather than at each call site means switching
    models touches ``agent/config.py`` alone.
    """
    name = (provider or config.PROVIDER).strip().lower()
    if name == "gemini":
        return GeminiLLMClient(**overrides)
    if name == "claude":
        return ClaudeLLMClient(**overrides)
    raise LLMError(f"unknown LLM provider {name!r}; expected 'gemini' or 'claude'")


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
