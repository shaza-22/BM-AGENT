"""Tests for the LLM interface and its two implementations.

No network and no API key: the Claude client is exercised against a stub that
records the request it was handed.
"""

from __future__ import annotations

import types

import pytest

from agent.llm import ClaudeLLMClient, FakeLLMClient, LLMError


def text_response(text: str, *, stop_reason: str = "end_turn"):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
    )


class StubAnthropic:
    """Minimal stand-in exposing both the beta and non-beta create paths."""

    def __init__(self, response=None, beta_error: Exception | None = None):
        self.response = response or text_response("ok")
        self.beta_error = beta_error
        self.beta_calls: list[dict] = []
        self.calls: list[dict] = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return outer.response

        class _BetaMessages:
            def create(self, **kwargs):
                outer.beta_calls.append(kwargs)
                if outer.beta_error:
                    raise outer.beta_error
                return outer.response

        self.messages = _Messages()
        self.beta = types.SimpleNamespace(messages=_BetaMessages())


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
            ClaudeLLMClient(client=stub).complete("hi")

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
