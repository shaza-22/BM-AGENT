"""Tests for the LLM interface and its two implementations.

No network and no API key: the Claude client is exercised against a stub that
records the request it was handed.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from conftest import FAKE_KEY, StubAnthropic, StubGemini, no_sleep, text_response

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
        stub = StubGemini(text='{"choice": 1}', error=ValueError("Invalid JSON schema supplied"),
                          errors_until=1)
        client = GeminiLLMClient(client=stub)
        assert client.complete("hi", schema={"type": "object"}) == '{"choice": 1}'
        assert "response_json_schema" in stub.configs[0]
        assert "response_json_schema" not in stub.configs[1]

    def test_schema_rejection_is_remembered(self):
        stub = StubGemini(text="{}", error=ValueError("Invalid JSON schema supplied"),
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
        stub = StubGemini(text="{}", error=ValueError("Invalid JSON schema supplied"),
                          errors_until=1)
        client = GeminiLLMClient(client=stub, sleep=no_sleep)
        client.complete("hi", schema={"type": "object"})
        assert len(stub.configs) == 2    # one rejected, one without the schema
