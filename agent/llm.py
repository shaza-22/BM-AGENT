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
import time
from typing import Any, Callable, Protocol, Sequence

from agent import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or refuses to answer.

    The navigator catches this and ends the run with ``status="error"`` rather
    than letting an API problem crash a task mid-way. ``status`` carries the
    HTTP code when one could be determined, and ``retryable`` records whether
    the failure was the kind that retrying could have fixed.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        quota_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        # Seconds the provider asked us to wait, when it said so.
        self.retry_after = retry_after
        self.quota_ids = quota_ids


def _status_of(exc: Exception) -> int | None:
    """Best-effort HTTP status for an SDK exception.

    Each SDK names it differently -- anthropic uses ``status_code``,
    google-genai uses ``code`` -- and some transports only put it in the
    message, so all three are tried.
    """
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response_status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(response_status, int):
        return response_status
    match = re.search(r"\b([45]\d{2})\b", str(exc))
    return int(match.group(1)) if match else None


def _error_details(exc: Exception) -> list[dict]:
    """The google.rpc detail objects attached to an API error, if any.

    Read structurally rather than by matching the message: the human-readable
    text is unstable and, for quota failures, does not name the quota at all.
    """
    payload = getattr(exc, "details", None)
    if isinstance(payload, dict):
        inner = payload.get("error", payload)
        payload = inner.get("details") if isinstance(inner, dict) else None
    if not isinstance(payload, list):
        return []
    return [detail for detail in payload if isinstance(detail, dict)]


def _quota_ids(exc: Exception) -> tuple[str, ...]:
    ids: list[str] = []
    for detail in _error_details(exc):
        if "QuotaFailure" not in str(detail.get("@type", "")):
            continue
        for violation in detail.get("violations") or []:
            if isinstance(violation, dict) and violation.get("quotaId"):
                ids.append(str(violation["quotaId"]))
    if not ids:
        # Some transports fold the payload into the message; the field is still
        # named there, so this reads the same field rather than guessing.
        ids = re.findall(r'"quotaId"\s*:\s*"([^"]+)"', str(exc))
    return tuple(ids)


def _retry_after(exc: Exception) -> float | None:
    """The delay a RetryInfo detail asked for, in seconds."""
    for detail in _error_details(exc):
        if "RetryInfo" not in str(detail.get("@type", "")):
            continue
        match = re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", "")).strip())
        if match:
            return float(match.group(1))
    match = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', str(exc))
    return float(match.group(1)) if match else None


def _is_daily_quota(quota_id: str) -> bool:
    # Separators are stripped from both sides so the same marker matches
    # "PerDay", "per_day" and "per-day".
    lowered = re.sub(r"[-_\s]", "", quota_id.lower())
    return any(
        re.sub(r"[-_\s]", "", marker.lower()) in lowered
        for marker in config.LLM_DAILY_QUOTA_MARKERS
    )


def _looks_transient(exc: Exception) -> bool:
    """Whether a status-less failure is the kind a retry could fix.

    Deliberately narrow: a connection reset or an overload notice is worth
    another attempt, a TypeError in our own request is not.
    """
    name = getattr(getattr(exc, "status", None), "name", None) or getattr(exc, "status", None)
    if isinstance(name, str) and name.strip().upper() in config.LLM_TRANSIENT_STATUS_NAMES:
        return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in haystack
        for marker in ("timeout", "timed out", "connection", "unavailable", "overloaded", "temporarily")
    )


def _as_llm_error(exc: Exception, provider: str) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    status = _status_of(exc)
    quota_ids = _quota_ids(exc)

    # A per-day quota does not refill in seconds, so backing off inside a run
    # only spends time to fail anyway. Per-minute quotas are worth waiting out.
    daily = [quota for quota in quota_ids if _is_daily_quota(quota)]
    if daily:
        return LLMError(
            f"{provider} quota exhausted for the day (quotaId={daily[0]}). Retrying will "
            f"not help: this quota refills on the provider's daily schedule, not in "
            f"seconds. Wait for the reset, raise the quota, or switch model or provider.",
            status=status,
            retryable=False,
            quota_ids=quota_ids,
        )

    if status in config.LLM_PERMANENT_STATUS:
        retryable = False
    elif status in config.LLM_TRANSIENT_STATUS:
        retryable = True
    else:
        retryable = status is None and _looks_transient(exc)
    # `from None` at the raise site keeps the raw SDK error out of tracebacks;
    # its message is scrubbed of key-shaped text before it is carried over.
    return LLMError(
        f"{provider} request failed: {scrub(str(exc))}",
        status=status,
        retryable=retryable,
        retry_after=_retry_after(exc) if retryable else None,
        quota_ids=quota_ids,
    )


def retry_api_call(
    operation: Callable[[], Any],
    *,
    provider: str,
    attempts: int | None = None,
    sleep: Callable[[float], None] | None = None,
    on_retry: Callable[[float], None] | None = None,
) -> Any:
    """Run one API call, retrying transient failures with exponential backoff.

    Mirrors what ``browsing.fetcher`` already does for HTTP: overload and
    gateway failures get another attempt, settled answers such as an invalid
    key or an unknown model fail immediately. Every retry is logged, because a
    run that silently took four attempts is a run whose timing needs
    explaining.
    """
    total = max(1, attempts if attempts is not None else config.LLM_MAX_ATTEMPTS)
    pause = sleep if sleep is not None else time.sleep

    for attempt in range(1, total + 1):
        try:
            return operation()
        except Exception as exc:
            error = _as_llm_error(exc, provider)
            if not error.retryable or attempt == total:
                if error.retryable:
                    logger.error("%s call failed after %d attempts: %s", provider, total, error)
                raise error from None
            requested = error.retry_after
            delay = min(
                requested
                if requested is not None
                else config.LLM_RETRY_BACKOFF_S * (2 ** (attempt - 1)),
                config.LLM_RETRY_MAX_BACKOFF_S,
            )
            source = "server-requested" if requested is not None else "exponential"
            if requested is not None and requested > delay:
                source = f"server-requested {requested:.0f}s, capped"
            logger.warning(
                "%s call failed (status=%s, attempt %d/%d): %s -- retrying in %.1fs (%s)",
                provider, error.status, attempt, total, error, delay, source,
            )
            if on_retry is not None:
                on_retry(delay)
            pause(delay)

    raise LLMError(f"{provider} call exhausted its attempts")  # pragma: no cover


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


class _TimedClient:
    """Mixin: records retry waits and total time spent inside the API.

    The navigator reports these so a slow run can be attributed to the model,
    to backing off after a failure, or to page fetching -- rather than guessed
    at, or blamed on a rate limiter that does not exist at this layer.
    """

    retries: int
    retry_wait_s: float
    api_seconds: float

    def _record_retry(self, delay: float) -> None:
        self.retries += 1
        self.retry_wait_s += delay

    def _timed(self, operation: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            return operation()
        finally:
            self.api_seconds += time.perf_counter() - started


class ClaudeLLMClient(_TimedClient):
    """Claude-backed :class:`LLMClient`.

    ``schema`` is passed through as a structured-output constraint, so the
    reply is valid JSON matching it. The selector still parses defensively --
    a different ``LLMClient`` implementation may offer no such guarantee.

    Reasoning depth is controlled by ``effort`` (``CLAUDE_EFFORT``), the
    counterpart to Gemini's thinking budget. It defaults to ``"low"``, which is
    already the cheap setting for a task like link selection. Note that
    *disabling* thinking on this model is not the equivalent move and is not
    offered: with thinking off it can write a tool call into visible text or
    leak reasoning tags, so lowering effort is the supported way to spend less.
    """

    def __init__(
        self,
        *,
        model: str = config.CLAUDE_MODEL,
        effort: str = config.CLAUDE_EFFORT,
        max_tokens: int = config.CLAUDE_MAX_TOKENS,
        client: Any | None = None,
        enable_fallbacks: bool = True,
        max_attempts: int | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts if max_attempts is not None else config.LLM_MAX_ATTEMPTS
        self._sleep = sleep
        self._client = client
        self.retries = 0
        self.retry_wait_s = 0.0
        self.api_seconds = 0.0
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
        started = time.perf_counter()
        response = self._timed(lambda: self._send(client, request))
        logger.info(
            "llm call provider=Claude model=%s ms=%d effort=%s schema=%s",
            self.model, int((time.perf_counter() - started) * 1000), self.effort,
            "on" if schema is not None else "off",
        )

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
        """Send the request, degrading gracefully if fallbacks are unsupported.

        Transient failures are retried inside each branch, so an overloaded
        endpoint does not look like an unsupported parameter and trigger the
        degradation path by mistake.
        """
        if self._enable_fallbacks:
            try:
                return retry_api_call(
                    lambda: client.beta.messages.create(
                        betas=["server-side-fallback-2026-07-01"],
                        fallbacks="default",
                        **request,
                    ),
                    provider="Claude",
                    attempts=self.max_attempts,
                    sleep=self._sleep,
                    on_retry=self._record_retry,
                )
            except LLMError as exc:
                if not _looks_like_unsupported_parameter(exc):
                    raise
                # An older SDK or endpoint that does not know the parameter --
                # disable it for the rest of the run rather than failing.
                logger.warning("refusal fallbacks unsupported here (%s); continuing without", exc)
                self._enable_fallbacks = False

        return retry_api_call(
            lambda: client.messages.create(**request),
            provider="Claude",
            attempts=self.max_attempts,
            sleep=self._sleep,
            on_retry=self._record_retry,
        )


def _looks_like_unsupported_parameter(exc: Exception) -> bool:
    # A retryable failure is an overloaded endpoint, not an unknown parameter;
    # treating it as one would silently disable fallbacks for the whole run.
    if isinstance(exc, LLMError) and exc.retryable:
        return False
    message = str(exc).lower()
    return isinstance(exc, TypeError) or "beta" in message or "fallback" in message


class _OptionRejected(Exception):
    """Internal: the API would not accept one of the optional request settings.

    Both structured output and the thinking configuration are best-effort: they
    make the call better when supported, and must not fail it when not. The
    client drops whichever the API named and retries once.
    """

    def __init__(self, option: str | None, error: LLMError) -> None:
        super().__init__(str(error))
        # None when the API refused the request without naming a field, which
        # is what Google does: a bare "Request contains an invalid argument."
        self.option = option
        self.error = error


OPT_THINKING = "thinking"
OPT_SCHEMA = "schema"


class GeminiLLMClient(_TimedClient):
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
        thinking_budget: int | None = config.GEMINI_THINKING_BUDGET,
        thinking_level: str | None = config.GEMINI_THINKING_LEVEL,
        client: Any | None = None,
        api_key_env: str = config.GEMINI_API_KEY_ENV,
        max_attempts: int | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.max_attempts = max_attempts if max_attempts is not None else config.LLM_MAX_ATTEMPTS
        self._sleep = sleep
        self._client = client
        self._api_key_env = api_key_env
        self.retries = 0
        self.retry_wait_s = 0.0
        self.api_seconds = 0.0
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
        """Send one prompt, shedding optional settings the API will not accept.

        Structured output and the thinking configuration are best-effort: they
        improve the call where supported and must never cost it. Google refuses
        an unsupported setting with a bare 400 -- "Request contains an invalid
        argument." with no field named -- so which one is at fault cannot be
        read from the message. Instead they are dropped one at a time, in
        configured order, and the call retried after each.

        A setting is only disabled for the rest of the run if dropping it
        actually made the call succeed. If the request was malformed for some
        unrelated reason, every setting is restored and the original error is
        raised, so a bad request never silently degrades the run's quality.
        """
        client = self._ensure_client()
        self.calls += 1
        dropped: list[str] = []

        while True:
            try:
                result = self._timed(
                    lambda: self._generate(client, prompt, system, schema, frozenset(dropped))
                )
            except _OptionRejected as exc:
                candidate = exc.option or self._next_droppable(schema, dropped)
                if candidate is None or candidate in dropped:
                    # Nothing left to try; nothing was disabled along the way.
                    raise exc.error from None
                dropped.append(candidate)
                logger.warning(
                    "Gemini refused the request (%s) -- retrying without the %s setting%s",
                    exc.error, candidate,
                    "" if exc.option else " (the API named no field, so settings are shed in order)",
                )
                continue

            if dropped:
                # Only now is it known that these were the problem.
                for option in dropped:
                    self._disable(option)
                logger.warning(
                    "Gemini: %s disabled for the rest of the run", " and ".join(dropped)
                )
            return result

    def _in_use(self, option: str, schema: dict | None) -> bool:
        if option == OPT_THINKING:
            return self.thinking_budget is not None or self.thinking_level is not None
        if option == OPT_SCHEMA:
            return schema is not None and self._structured_output
        return False

    def _next_droppable(self, schema: dict | None, dropped: Sequence[str]) -> str | None:
        for option in config.LLM_OPTIONAL_SETTING_DROP_ORDER:
            if option not in dropped and self._in_use(option, schema):
                return option
        return None

    def _disable(self, option: str) -> None:
        if option == OPT_THINKING:
            self.thinking_budget = None
            self.thinking_level = None
        elif option == OPT_SCHEMA:
            self._structured_output = False

    def _generate(
        self,
        client: Any,
        prompt: str,
        system: str | None,
        schema: dict | None,
        dropped: frozenset[str] = frozenset(),
    ) -> str:
        # Passed as a plain dict rather than types.GenerateContentConfig: the
        # SDK accepts either, and this keeps the whole call path free of any
        # google.genai import, so an injected client works on a machine that
        # does not have the package at all.
        settings: dict[str, Any] = {"max_output_tokens": self.max_tokens}
        if system:
            settings["system_instruction"] = system

        used_schema = self._in_use(OPT_SCHEMA, schema) and OPT_SCHEMA not in dropped
        if used_schema:
            settings["response_mime_type"] = "application/json"
            settings["response_json_schema"] = schema

        thinking: dict[str, Any] = {}
        if OPT_THINKING in dropped:
            pass
        elif self.thinking_budget is not None or self.thinking_level is not None:
            if self.thinking_budget is not None:
                thinking["thinking_budget"] = self.thinking_budget
            if self.thinking_level is not None:
                thinking["thinking_level"] = self.thinking_level
        if thinking:
            settings["thinking_config"] = thinking

        started = time.perf_counter()
        try:
            response = retry_api_call(
                lambda: client.models.generate_content(
                    model=self.model, contents=prompt, config=settings
                ),
                provider="Gemini",
                attempts=self.max_attempts,
                sleep=self._sleep,
                on_retry=self._record_retry,
            )
        except LLMError as exc:
            # A refused setting comes back as a 400, which is never retried.
            # If any optional setting was sent, hand it to complete() to shed
            # -- whether or not the API said which one it disliked.
            if _is_bad_request(exc) and (used_schema or thinking):
                named = _named_option(str(exc), used_schema, bool(thinking))
                raise _OptionRejected(named, exc) from None
            raise

        logger.info(
            "llm call provider=Gemini model=%s ms=%d thinking=%s schema=%s",
            self.model, int((time.perf_counter() - started) * 1000),
            _thinking_label(thinking), "on" if used_schema else "off",
        )

        text = response.text
        if not text:
            raise LLMError(f"Gemini returned no text ({_finish_reason(response)})")
        return text


def _is_bad_request(exc: LLMError) -> bool:
    """Whether the provider rejected the request itself, rather than failing.

    Structural, not textual: Google answers an unsupported setting with a bare
    "Request contains an invalid argument." that names no field, so the status
    is the only reliable signal.
    """
    if exc.retryable:
        return False
    if exc.status == 400:
        return True
    lowered = str(exc).lower()
    return exc.status is None and ("invalid argument" in lowered or "invalid_argument" in lowered)


def _named_option(message: str, used_schema: bool, used_thinking: bool) -> str | None:
    """The setting the API named, when it named one -- a shortcut, not the rule.

    Only a setting that was actually sent can be blamed. When nothing is named,
    the caller sheds settings in order instead.
    """
    lowered = message.lower()
    if used_thinking and ("thinking" in lowered or "thought" in lowered):
        return OPT_THINKING
    if used_schema and ("schema" in lowered or "response_mime_type" in lowered):
        return OPT_SCHEMA
    return None


def _thinking_label(thinking: dict[str, Any]) -> str:
    if not thinking:
        return "default"
    if thinking.get("thinking_budget") == 0:
        return "off"
    return ",".join(f"{key.replace('thinking_', '')}={value}" for key, value in thinking.items())


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
