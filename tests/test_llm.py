"""Tests for the LLM interface and its two implementations.

No network and no API key: the Claude client is exercised against a stub that
records the request it was handed.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from conftest import (
    FAKE_KEY,
    StubAnthropic,
    StubGemini,
    api_error,
    bad_request,
    no_sleep,
    quota_error,
    text_response,
)

from agent import config
from agent.llm import (
    retry_api_call,
    ClaudeLLMClient,
    FakeLLMClient,
    GeminiLLMClient,
    LLMError,
    make_llm_client,
    resolve_api_key,
    scrub,
)



class TestFakeLLMClient:
    def test_scripted_responses_are_returned_in_order(self):
        llm = FakeLLMClient(["first", "second"])
        assert llm.complete("a") == "first"
        assert llm.complete("b") == "second"

    def test_last_response_repeats_once_exhausted(self):
        llm = FakeLLMClient(["only"])
        assert [llm.complete("a"), llm.complete("b")] == ["only", "only"]

    def test_callable_responses_see_the_prompt(self):
        llm = FakeLLMClient(lambda prompt: prompt.upper())
        assert llm.complete("hello") == "HELLO"

    def test_prompts_and_schemas_are_recorded(self):
        llm = FakeLLMClient(["x"])
        llm.complete("prompt", system="sys", schema={"type": "object"})
        assert llm.prompts == ["prompt"]
        assert llm.systems == ["sys"]
        assert llm.schemas == [{"type": "object"}]
        assert llm.calls == 1

    def test_empty_script_raises_rather_than_returning_nothing(self):
        with pytest.raises(LLMError):
            FakeLLMClient([]).complete("a")


class TestClaudeLLMClient:
    def test_request_shape(self):
        stub = StubAnthropic(text_response('{"choice": 0}'))
        client = ClaudeLLMClient(client=stub)
        result = client.complete("pick one", system="be brief", schema={"type": "object"})

        assert result == '{"choice": 0}'
        request = stub.beta_calls[0]
        assert request["model"] == "claude-opus-5"
        assert request["system"] == "be brief"
        assert request["messages"] == [{"role": "user", "content": "pick one"}]
        # effort keeps adaptive thinking short; the schema constrains the reply
        assert request["output_config"]["effort"] == "low"
        assert request["output_config"]["format"] == {
            "type": "json_schema",
            "schema": {"type": "object"},
        }

    def test_schema_is_omitted_when_not_requested(self):
        stub = StubAnthropic()
        ClaudeLLMClient(client=stub).complete("hi")
        assert "format" not in stub.beta_calls[0]["output_config"]
        assert "system" not in stub.beta_calls[0]

    def test_refusal_fallbacks_are_requested_by_default(self):
        stub = StubAnthropic()
        ClaudeLLMClient(client=stub).complete("hi")
        assert stub.beta_calls[0]["fallbacks"] == "default"
        assert stub.beta_calls[0]["betas"] == ["server-side-fallback-2026-07-01"]

    def test_unsupported_fallbacks_degrade_instead_of_failing(self):
        # An older SDK that does not know the parameter must not break the run.
        stub = StubAnthropic(beta_error=TypeError("unexpected keyword argument 'fallbacks'"))
        client = ClaudeLLMClient(client=stub)
        assert client.complete("hi") == "ok"
        assert len(stub.calls) == 1

    def test_degradation_is_remembered_for_later_calls(self):
        stub = StubAnthropic(beta_error=TypeError("no fallbacks"))
        client = ClaudeLLMClient(client=stub)
        client.complete("one")
        client.complete("two")
        assert len(stub.beta_calls) == 1  # not retried on the second call
        assert len(stub.calls) == 2

    def test_fallbacks_can_be_disabled(self):
        stub = StubAnthropic()
        ClaudeLLMClient(client=stub, enable_fallbacks=False).complete("hi")
        assert stub.beta_calls == []
        assert len(stub.calls) == 1

    def test_refusal_becomes_an_llm_error(self):
        stub = StubAnthropic(text_response("", stop_reason="refusal"))
        with pytest.raises(LLMError, match="refused"):
            ClaudeLLMClient(client=stub).complete("hi")

    def test_api_failure_becomes_an_llm_error(self):
        stub = StubAnthropic(beta_error=RuntimeError("connection reset"))
        with pytest.raises(LLMError, match="connection reset"):
            ClaudeLLMClient(client=stub, sleep=no_sleep).complete("hi")

    def test_missing_sdk_reports_how_to_install_it(self, monkeypatch):
        # The project must import and test without the anthropic package.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(LLMError, match="pip install anthropic"):
            ClaudeLLMClient().complete("hi")

    def test_model_and_effort_are_configurable(self):
        stub = StubAnthropic()
        ClaudeLLMClient(client=stub, model="claude-sonnet-5", effort="medium").complete("hi")
        assert stub.beta_calls[0]["model"] == "claude-sonnet-5"
        assert stub.beta_calls[0]["output_config"]["effort"] == "medium"


class TestGeminiLLMClient:
    def test_request_shape(self):
        stub = StubGemini('{"choice": 0}')
        client = GeminiLLMClient(client=stub, model="gemini-2.5-flash")
        result = client.complete("pick one", system="be brief", schema={"type": "object"})

        assert result == '{"choice": 0}'
        assert stub.last_model == "gemini-2.5-flash"
        assert stub.last_contents == "pick one"
        settings = stub.configs[0]
        assert settings["system_instruction"] == "be brief"
        assert settings["response_mime_type"] == "application/json"
        assert settings["response_json_schema"] == {"type": "object"}
        assert settings["max_output_tokens"] == config.GEMINI_MAX_TOKENS

    def test_schema_is_omitted_when_not_requested(self):
        stub = StubGemini()
        GeminiLLMClient(client=stub).complete("hi")
        assert "response_json_schema" not in stub.configs[0]
        assert "response_mime_type" not in stub.configs[0]

    def test_schema_rejection_falls_back_to_defensive_parsing(self):
        # Structured output is a different binding on this API, so a rejection
        # must degrade to plain text rather than failing every hop.
        stub = StubGemini(text='{"choice": 1}', error=bad_request("Invalid JSON schema supplied"),
                          errors_until=1)
        client = GeminiLLMClient(client=stub)
        assert client.complete("hi", schema={"type": "object"}) == '{"choice": 1}'
        assert "response_json_schema" in stub.configs[0]
        assert "response_json_schema" not in stub.configs[1]

    def test_schema_rejection_is_remembered(self):
        stub = StubGemini(text="{}", error=bad_request("Invalid JSON schema supplied"),
                          errors_until=1)
        client = GeminiLLMClient(client=stub)
        client.complete("one", schema={"type": "object"})
        client.complete("two", schema={"type": "object"})
        # One rejected attempt, then three schema-free calls.
        assert sum(1 for c in stub.configs if "response_json_schema" in c) == 1

    def test_other_errors_become_llm_errors(self):
        stub = StubGemini(error=RuntimeError("429 quota exceeded"))
        with pytest.raises(LLMError, match="quota exceeded"):
            GeminiLLMClient(client=stub, sleep=no_sleep).complete("hi")

    def test_empty_reply_reports_why(self):
        stub = StubGemini(text="", candidates=[types.SimpleNamespace(finish_reason="SAFETY")])
        with pytest.raises(LLMError, match="SAFETY"):
            GeminiLLMClient(client=stub).complete("hi")

    def test_missing_sdk_reports_how_to_install_it(self, monkeypatch):
        # The project must import and test without google-genai installed.
        # A None entry in sys.modules makes the import raise, which is what an
        # absent package looks like from inside _ensure_client.
        import google

        monkeypatch.setitem(sys.modules, "google.genai", None)
        monkeypatch.delattr(google, "genai", raising=False)
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, FAKE_KEY)
        with pytest.raises(LLMError, match="pip install google-genai"):
            GeminiLLMClient().complete("hi")


class TestApiKeyHandling:
    def test_missing_key_names_the_variable_not_a_raw_sdk_error(self, monkeypatch):
        monkeypatch.setattr("agent.llm.load_project_env", lambda: None)
        monkeypatch.delenv(config.GEMINI_API_KEY_ENV, raising=False)
        with pytest.raises(LLMError) as excinfo:
            GeminiLLMClient().complete("hi")
        message = str(excinfo.value)
        assert config.GEMINI_API_KEY_ENV in message
        assert ".env" in message

    def test_blank_key_is_treated_as_missing(self, monkeypatch):
        monkeypatch.setattr("agent.llm.load_project_env", lambda: None)
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, "   ")
        with pytest.raises(LLMError, match="no Gemini API key found"):
            resolve_api_key(config.GEMINI_API_KEY_ENV, "Gemini")

    def test_key_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setattr("agent.llm.load_project_env", lambda: None)
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, FAKE_KEY)
        assert resolve_api_key(config.GEMINI_API_KEY_ENV, "Gemini") == FAKE_KEY

    def test_key_never_appears_in_an_error_message(self, monkeypatch, caplog):
        # SDK errors sometimes echo request details, and an error message is
        # exactly the text that ends up pasted into a bug report.
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, FAKE_KEY)
        stub = StubGemini(error=RuntimeError(f"401 from ?key={FAKE_KEY}"))
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMError) as excinfo:
                GeminiLLMClient(client=stub).complete("hi")

        assert FAKE_KEY not in str(excinfo.value)
        assert "<redacted>" in str(excinfo.value)
        assert FAKE_KEY not in caplog.text

    def test_the_raw_sdk_error_is_not_chained_into_the_traceback(self, monkeypatch):
        # A chained __cause__ would reproduce the unscrubbed message wherever
        # the traceback is printed.
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, FAKE_KEY)
        stub = StubGemini(error=RuntimeError(f"boom {FAKE_KEY}"))
        with pytest.raises(LLMError) as excinfo:
            GeminiLLMClient(client=stub).complete("hi")
        assert excinfo.value.__cause__ is None

    def test_key_is_not_stored_on_the_client_or_its_repr(self, monkeypatch):
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, FAKE_KEY)
        client = GeminiLLMClient(client=StubGemini())
        client.complete("hi")
        assert FAKE_KEY not in repr(client)
        assert not any(FAKE_KEY == str(value) for value in vars(client).values())

    @pytest.mark.parametrize(
        "message",
        [
            f"error with {FAKE_KEY}",
            "https://api/v1?key=AIzaSyABCDEFGHIJKLMNOP123&alt=json",
            "authorization failed for sk-ant-api03-ABCDEFGHIJKLMNOPQRS",
        ],
    )
    def test_scrub_removes_key_shaped_text(self, message):
        assert "<redacted>" in scrub(message)
        assert "AIza" not in scrub(message) or "AIza" not in message
        assert "sk-ant-api03-ABCDEFGHIJKLMNOPQRS" not in scrub(message)

    def test_dotenv_is_optional(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dotenv":
                raise ImportError("No module named 'dotenv'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, FAKE_KEY)
        # The key is already in the environment, so a missing .env loader is fine.
        assert resolve_api_key(config.GEMINI_API_KEY_ENV, "Gemini") == FAKE_KEY


class TestProviderSelection:
    def test_default_provider_comes_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "PROVIDER", "gemini")
        assert isinstance(make_llm_client(), GeminiLLMClient)
        monkeypatch.setattr(config, "PROVIDER", "claude")
        assert isinstance(make_llm_client(), ClaudeLLMClient)

    def test_provider_can_be_named_explicitly(self):
        assert isinstance(make_llm_client("claude"), ClaudeLLMClient)
        assert isinstance(make_llm_client("GEMINI"), GeminiLLMClient)

    def test_overrides_reach_the_client(self):
        assert make_llm_client("gemini", model="gemini-2.5-pro").model == "gemini-2.5-pro"

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(LLMError, match="unknown LLM provider"):
            make_llm_client("gpt")

    def test_building_a_client_does_not_need_a_key(self, monkeypatch):
        # Keys are resolved at connect time, so constructing a Navigator is
        # safe in a test or on a machine with no credentials.
        monkeypatch.setattr("agent.llm.load_project_env", lambda: None)
        monkeypatch.delenv(config.GEMINI_API_KEY_ENV, raising=False)
        assert make_llm_client("gemini") is not None


class TestTransientRetry:
    """A single 503 from a free-tier endpoint used to end a whole run.

    The fetcher already retried transient HTTP failures; the selector is called
    once per hop, so the same failure there was far more expensive.
    """

    def recording_sleep(self):
        delays: list[float] = []
        return delays, delays.append

    @pytest.mark.parametrize("status", sorted(config.LLM_TRANSIENT_STATUS))
    def test_transient_statuses_are_retried(self, status):
        calls = {"n": 0}

        def operation():
            calls["n"] += 1
            if calls["n"] < 3:
                error = RuntimeError(f"{status} transient")
                error.code = status
                raise error
            return "ok"

        assert retry_api_call(operation, provider="Test", sleep=no_sleep) == "ok"
        assert calls["n"] == 3

    @pytest.mark.parametrize("status", sorted(config.LLM_PERMANENT_STATUS))
    def test_permanent_statuses_fail_fast(self, status):
        # A bad key or an unknown model will not fix itself; retrying only
        # delays a clear message.
        calls = {"n": 0}

        def operation():
            calls["n"] += 1
            error = RuntimeError(f"{status} not happening")
            error.code = status
            raise error

        with pytest.raises(LLMError) as excinfo:
            retry_api_call(operation, provider="Test", sleep=no_sleep)
        assert calls["n"] == 1
        assert excinfo.value.status == status
        assert excinfo.value.retryable is False

    def test_backoff_is_exponential_and_capped(self):
        delays, record = self.recording_sleep()

        def always_503():
            error = RuntimeError("503 UNAVAILABLE")
            error.code = 503
            raise error

        with pytest.raises(LLMError):
            retry_api_call(always_503, provider="Test", attempts=5, sleep=record)

        assert delays == [2.0, 4.0, 8.0, 16.0]
        assert all(delay <= config.LLM_RETRY_MAX_BACKOFF_S for delay in delays)

    def test_attempt_count_is_configurable(self):
        calls = {"n": 0}

        def always_503():
            calls["n"] += 1
            error = RuntimeError("503 UNAVAILABLE")
            error.code = 503
            raise error

        with pytest.raises(LLMError):
            retry_api_call(always_503, provider="Test", attempts=2, sleep=no_sleep)
        assert calls["n"] == 2

    def test_every_retry_is_logged(self, caplog):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                error = RuntimeError("503 UNAVAILABLE")
                error.code = 503
                raise error
            return "ok"

        with caplog.at_level(logging.WARNING):
            retry_api_call(flaky, provider="Gemini", sleep=no_sleep)
        assert len(caplog.records) == 2
        assert all("retrying in" in record.getMessage() for record in caplog.records)
        assert "status=503" in caplog.text
        assert "Gemini" in caplog.text

    def test_google_style_status_name_without_a_code_is_retried(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                error = RuntimeError("service is having trouble")
                error.status = "UNAVAILABLE"
                raise error
            return "ok"

        assert retry_api_call(flaky, provider="Gemini", sleep=no_sleep) == "ok"

    def test_connection_failures_without_a_status_are_retried(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("connection reset by peer")
            return "ok"

        assert retry_api_call(flaky, provider="Gemini", sleep=no_sleep) == "ok"

    def test_programming_errors_are_not_retried(self):
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise TypeError("unexpected keyword argument")

        with pytest.raises(LLMError):
            retry_api_call(broken, provider="Test", sleep=no_sleep)
        assert calls["n"] == 1

    def test_retry_message_is_scrubbed(self):
        def leaky():
            error = RuntimeError(f"503 from ?key={FAKE_KEY}")
            error.code = 503
            raise error

        with pytest.raises(LLMError) as excinfo:
            retry_api_call(leaky, provider="Gemini", attempts=2, sleep=no_sleep)
        assert FAKE_KEY not in str(excinfo.value)


class TestClientsRetry:
    def test_gemini_recovers_from_a_transient_failure(self):
        # The live symptom: 503 UNAVAILABLE on hop 1 ended the run at 0 hops.
        error = RuntimeError("503 UNAVAILABLE")
        error.code = 503
        stub = StubGemini(text='{"choice": 0}', error=error, errors_until=2)
        client = GeminiLLMClient(client=stub, sleep=no_sleep)
        assert client.complete("hi") == '{"choice": 0}'
        assert len(stub.configs) == 3

    def test_claude_recovers_from_a_transient_failure(self):
        error = RuntimeError("529 overloaded")
        error.status_code = 503
        stub = StubAnthropic(text_response("ok"), beta_error=error, errors_until=1)
        client = ClaudeLLMClient(client=stub, sleep=no_sleep)
        assert client.complete("hi") == "ok"

    def test_an_overloaded_endpoint_does_not_disable_claude_fallbacks(self):
        # A retryable failure must not be mistaken for an unsupported
        # parameter, which would silently drop fallbacks for the whole run.
        error = RuntimeError("503 overloaded")
        error.status_code = 503
        stub = StubAnthropic(text_response("ok"), beta_error=error, errors_until=1)
        client = ClaudeLLMClient(client=stub, sleep=no_sleep)
        client.complete("hi")
        assert client._enable_fallbacks is True
        assert stub.calls == []          # never fell through to the non-beta path

    def test_gemini_schema_rejection_is_still_not_retried(self):
        stub = StubGemini(text="{}", error=bad_request("Invalid JSON schema supplied"),
                          errors_until=1)
        client = GeminiLLMClient(client=stub, sleep=no_sleep)
        client.complete("hi", schema={"type": "object"})
        assert len(stub.configs) == 2    # one rejected, one without the schema


class TestTimingVisibility:
    def test_time_spent_in_the_api_is_recorded(self):
        client = GeminiLLMClient(client=StubGemini("ok"))
        client.complete("hi")
        assert client.api_seconds > 0.0
        assert client.calls == 1

    def test_retry_waits_are_recorded_separately(self):
        error = RuntimeError("503 UNAVAILABLE")
        error.code = 503
        stub = StubGemini(text="ok", error=error, errors_until=2)
        client = GeminiLLMClient(client=stub, sleep=no_sleep)
        client.complete("hi")
        # Two backoffs at 2s and 4s: reported, not silently absorbed into the
        # call time, so a slow hop can be blamed on the right thing.
        assert client.retries == 2
        assert client.retry_wait_s == 6.0

    def test_claude_records_the_same_figures(self):
        client = ClaudeLLMClient(client=StubAnthropic(text_response("ok")))
        client.complete("hi")
        assert client.api_seconds > 0.0
        assert client.retries == 0
        assert client.retry_wait_s == 0.0


class TestThinkingConfiguration:
    """Reasoning depth is the dominant cost of a hop.

    Measured on a two-hop run at identical prompt sizes: 3.5s for one call
    against 32.8s for the other, 35.6s of a 38.7s run inside the model.
    """

    def test_the_budget_is_sent_when_configured(self):
        stub = StubGemini("ok")
        GeminiLLMClient(client=stub, thinking_budget=0, thinking_level=None).complete("hi")
        assert stub.configs[0]["thinking_config"] == {"thinking_budget": 0}

    def test_the_level_is_sent_when_configured(self):
        # Newer models take a level rather than a token budget.
        stub = StubGemini("ok")
        GeminiLLMClient(client=stub, thinking_budget=None, thinking_level="low").complete("hi")
        assert stub.configs[0]["thinking_config"] == {"thinking_level": "low"}

    def test_both_can_be_sent_together(self):
        stub = StubGemini("ok")
        GeminiLLMClient(client=stub, thinking_budget=0, thinking_level="low").complete("hi")
        assert stub.configs[0]["thinking_config"] == {"thinking_budget": 0, "thinking_level": "low"}

    def test_nothing_is_sent_when_unset(self):
        stub = StubGemini("ok")
        GeminiLLMClient(client=stub, thinking_budget=None, thinking_level=None).complete("hi")
        assert "thinking_config" not in stub.configs[0]

    def test_the_defaults_come_from_config(self):
        stub = StubGemini("ok")
        GeminiLLMClient(client=stub).complete("hi")
        sent = stub.configs[0].get("thinking_config", {})
        assert sent.get("thinking_budget") == config.GEMINI_THINKING_BUDGET
        assert sent.get("thinking_level") == config.GEMINI_THINKING_LEVEL

    def test_a_rejected_thinking_config_degrades_instead_of_failing(self, caplog):
        # Same contract as the schema fallback: an unsupported setting must not
        # cost the call, only the speedup.
        stub = StubGemini(text="ok", error=bad_request("thinking_budget is not supported"),
                          errors_until=1)
        client = GeminiLLMClient(client=stub, thinking_budget=0, thinking_level=None,
                                 sleep=no_sleep)
        with caplog.at_level(logging.WARNING):
            assert client.complete("hi") == "ok"
        assert "thinking_config" in stub.configs[0]
        assert "thinking_config" not in stub.configs[1]
        assert "without the thinking setting" in caplog.text
        assert "thinking disabled for the rest of the run" in caplog.text

    def test_a_rejected_thinking_config_is_remembered(self):
        stub = StubGemini(text="ok", error=bad_request("thinking_level unsupported"), errors_until=1)
        client = GeminiLLMClient(client=stub, thinking_budget=0, sleep=no_sleep)
        client.complete("one")
        client.complete("two")
        assert client.thinking_budget is None
        assert sum(1 for c in stub.configs if "thinking_config" in c) == 1

    def test_schema_and_thinking_degrade_independently(self):
        stub = StubGemini(text="ok", error=bad_request("Invalid JSON schema supplied"),
                          errors_until=1)
        client = GeminiLLMClient(client=stub, thinking_budget=0, sleep=no_sleep)
        client.complete("hi", schema={"type": "object"})
        # The schema was dropped; the thinking config was not.
        assert "response_json_schema" not in stub.configs[1]
        assert "thinking_config" in stub.configs[1]
        assert client.thinking_budget == 0

    def test_an_unrelated_400_is_not_blamed_on_an_option(self):
        stub = StubGemini(text="ok", error=bad_request("contents must not be empty"))
        client = GeminiLLMClient(client=stub, thinking_budget=0, sleep=no_sleep)
        with pytest.raises(LLMError):
            client.complete("hi", schema={"type": "object"})
        assert client.thinking_budget == 0        # nothing silently disabled
        assert client._structured_output is True

    def test_a_transient_failure_is_not_mistaken_for_a_rejected_option(self):
        stub = StubGemini(
            text="ok", error=api_error("503 UNAVAILABLE thinking service", code=503), errors_until=1
        )
        client = GeminiLLMClient(client=stub, thinking_budget=0, sleep=no_sleep)
        client.complete("hi")
        assert client.thinking_budget == 0        # retried, not degraded
        assert client.retries == 1


class TestPerCallLogging:
    def test_gemini_logs_model_duration_and_settings(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.llm"):
            GeminiLLMClient(client=StubGemini("ok"), thinking_budget=0).complete(
                "hi", schema={"type": "object"}
            )
        assert "llm call provider=Gemini" in caplog.text
        assert "thinking=off" in caplog.text
        assert "schema=on" in caplog.text
        assert "ms=" in caplog.text

    def test_gemini_reports_the_default_when_no_budget_is_set(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.llm"):
            GeminiLLMClient(
                client=StubGemini("ok"), thinking_budget=None, thinking_level=None
            ).complete("hi")
        assert "thinking=default" in caplog.text

    def test_claude_logs_the_same_shape_with_its_own_knob(self, caplog):
        # Both providers report per-call timing the same way; only the name of
        # the reasoning knob differs.
        with caplog.at_level(logging.INFO, logger="agent.llm"):
            ClaudeLLMClient(client=StubAnthropic(text_response("ok"))).complete("hi")
        assert "llm call provider=Claude" in caplog.text
        assert f"effort={config.CLAUDE_EFFORT}" in caplog.text
        assert "ms=" in caplog.text


class TestBareBadRequestDegradation:
    """Google refuses an unsupported setting without naming it.

    The live symptom: a thinking_budget rejection came back as
    "Request contains an invalid argument." and ended the run with
    status="error" instead of dropping the setting and retrying.
    """

    def test_a_bare_400_sheds_settings_instead_of_failing(self, caplog):
        stub = StubGemini(text="ok", error=bad_request(), errors_until=1)
        client = GeminiLLMClient(client=stub, thinking_level="low", sleep=no_sleep)
        with caplog.at_level(logging.WARNING):
            assert client.complete("hi") == "ok"
        assert "thinking_config" in stub.configs[0]
        assert "thinking_config" not in stub.configs[1]
        assert "named no field" in caplog.text

    def test_settings_are_shed_in_configured_order(self):
        # Thinking first: it is the more likely culprit and the cheaper loss.
        stub = StubGemini(text="ok", error=bad_request(), errors_until=1)
        client = GeminiLLMClient(client=stub, thinking_level="low", sleep=no_sleep)
        client.complete("hi", schema={"type": "object"})
        assert "thinking_config" not in stub.configs[1]
        assert "response_json_schema" in stub.configs[1]   # schema survived

    def test_a_second_bare_400_sheds_the_next_setting(self):
        stub = StubGemini(text="ok", error=bad_request(), errors_until=2)
        client = GeminiLLMClient(client=stub, thinking_level="low", sleep=no_sleep)
        client.complete("hi", schema={"type": "object"})
        assert len(stub.configs) == 3
        assert "thinking_config" not in stub.configs[2]
        assert "response_json_schema" not in stub.configs[2]

    def test_a_genuinely_bad_request_still_fails(self):
        stub = StubGemini(text="ok", error=bad_request("contents must not be empty"))
        client = GeminiLLMClient(client=stub, thinking_level="low", sleep=no_sleep)
        with pytest.raises(LLMError, match="must not be empty"):
            client.complete("hi", schema={"type": "object"})

    def test_nothing_is_disabled_when_shedding_did_not_help(self):
        # A setting is only given up if dropping it actually fixed the call, so
        # an unrelated bad request never quietly degrades the rest of the run.
        stub = StubGemini(text="ok", error=bad_request("contents must not be empty"))
        client = GeminiLLMClient(client=stub, thinking_level="low", sleep=no_sleep)
        with pytest.raises(LLMError):
            client.complete("hi", schema={"type": "object"})
        assert client.thinking_level == "low"
        assert client._structured_output is True

    def test_a_400_with_no_optional_settings_sent_fails_immediately(self):
        stub = StubGemini(text="ok", error=bad_request())
        client = GeminiLLMClient(
            client=stub, thinking_budget=None, thinking_level=None, sleep=no_sleep
        )
        with pytest.raises(LLMError):
            client.complete("hi")
        assert len(stub.configs) == 1

    def test_shedding_is_not_a_retry(self):
        # Dropping a setting is a different request, not the same one again --
        # it must not be counted as, or delayed like, a transient retry.
        stub = StubGemini(text="ok", error=bad_request(), errors_until=1)
        client = GeminiLLMClient(client=stub, thinking_level="low", sleep=no_sleep)
        client.complete("hi")
        assert client.retries == 0
        assert client.retry_wait_s == 0.0


class TestQuotaHandling:
    """A per-day quota does not refill in seconds.

    The live symptom: exhausting GenerateRequestsPerDayPerProjectPerModel burned
    14s across four attempts and failed anyway.
    """

    def test_a_daily_quota_fails_immediately(self):
        stub = StubGemini(error=quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        client = GeminiLLMClient(client=stub, sleep=no_sleep)
        with pytest.raises(LLMError) as excinfo:
            client.complete("hi")

        assert len(stub.configs) == 1          # no retries at all
        assert client.retries == 0
        assert client.retry_wait_s == 0.0
        assert excinfo.value.retryable is False

    def test_the_daily_message_says_what_to_do(self):
        stub = StubGemini(error=quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        with pytest.raises(LLMError) as excinfo:
            GeminiLLMClient(client=stub, sleep=no_sleep).complete("hi")
        message = str(excinfo.value)
        assert "exhausted for the day" in message
        assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in message
        assert "Retrying will not help" in message

    def test_a_per_minute_quota_is_still_retried(self):
        stub = StubGemini(
            text="ok",
            error=quota_error("GenerateRequestsPerMinutePerProjectPerModel"),
            errors_until=1,
        )
        client = GeminiLLMClient(client=stub, sleep=no_sleep)
        assert client.complete("hi") == "ok"
        assert client.retries == 1

    def test_the_server_requested_delay_is_honoured(self):
        delays: list[float] = []
        stub = StubGemini(
            text="ok",
            error=quota_error("GenerateRequestsPerMinutePerProjectPerModel", retry_delay="27s"),
            errors_until=1,
        )
        client = GeminiLLMClient(client=stub, sleep=delays.append)
        client.complete("hi")
        assert delays == [27.0]                # not the 2s exponential default

    def test_a_server_delay_beyond_the_cap_is_capped(self):
        delays: list[float] = []
        stub = StubGemini(
            text="ok",
            error=quota_error("GenerateRequestsPerMinutePerProjectPerModel", retry_delay="600s"),
            errors_until=1,
        )
        GeminiLLMClient(client=stub, sleep=delays.append).complete("hi")
        assert delays == [config.LLM_RETRY_MAX_BACKOFF_S]

    def test_the_retry_log_says_where_the_delay_came_from(self, caplog):
        stub = StubGemini(
            text="ok",
            error=quota_error("GenerateRequestsPerMinutePerProjectPerModel", retry_delay="5s"),
            errors_until=1,
        )
        with caplog.at_level(logging.WARNING):
            GeminiLLMClient(client=stub, sleep=no_sleep).complete("hi")
        assert "server-requested" in caplog.text

    @pytest.mark.parametrize(
        "quota_id",
        [
            "GenerateRequestsPerDayPerProjectPerModel",
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "generate_requests_per_day_per_project",
            "SomethingDailyLimit",
        ],
    )
    def test_daily_quota_ids_are_recognised(self, quota_id):
        stub = StubGemini(error=quota_error(quota_id))
        with pytest.raises(LLMError, match="exhausted for the day"):
            GeminiLLMClient(client=stub, sleep=no_sleep).complete("hi")

    def test_a_429_without_quota_details_is_still_retried(self):
        stub = StubGemini(text="ok", error=api_error("429 RESOURCE_EXHAUSTED", code=429),
                          errors_until=1)
        client = GeminiLLMClient(client=stub, sleep=no_sleep)
        assert client.complete("hi") == "ok"
        assert client.retries == 1

    def test_quota_ids_are_carried_on_the_error(self):
        stub = StubGemini(error=quota_error("GenerateRequestsPerDayPerProjectPerModel"))
        with pytest.raises(LLMError) as excinfo:
            GeminiLLMClient(client=stub, sleep=no_sleep).complete("hi")
        assert excinfo.value.quota_ids == ("GenerateRequestsPerDayPerProjectPerModel",)

    def test_quota_details_folded_into_the_message_are_still_read(self):
        # Some transports drop the structured payload and keep only the text;
        # the field is still named there.
        error = api_error(
            '429 RESOURCE_EXHAUSTED. {"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}',
            code=429,
        )
        stub = StubGemini(error=error)
        with pytest.raises(LLMError, match="exhausted for the day"):
            GeminiLLMClient(client=stub, sleep=no_sleep).complete("hi")
