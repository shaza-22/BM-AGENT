"""Provider-agnostic runtime reasoning backend interface and implementations."""

from abc import ABC, abstractmethod
import json
from typing import Any, Callable, Dict, List, Optional

from person_b.config import PersonBConfig, default_config
from person_b.errors import AdapterError


class ReasoningBackend(ABC):
    """Abstract interface for LLM reasoning backends."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion for a given prompt."""
        pass

    @abstractmethod
    def structured_complete(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a structured response adhering to a given JSON schema."""
        pass


class FakeReasoningBackend(ReasoningBackend):
    """In-memory deterministic reasoning backend for unit testing and offline development."""

    def __init__(self) -> None:
        self.canned_responses: Dict[str, str] = {}
        self.canned_structured: Dict[str, Dict[str, Any]] = {}
        self.completion_handler: Optional[Callable[[str, Optional[str]], str]] = None
        self.structured_handler: Optional[Callable[[str, Dict[str, Any], Optional[str]], Dict[str, Any]]] = None
        self.call_history: List[Dict[str, Any]] = []

    def register_response(self, prompt_substring: str, response: str) -> None:
        """Register a canned text response for prompts containing the substring."""
        self.canned_responses[prompt_substring] = response

    def register_structured(self, prompt_substring: str, response: Dict[str, Any]) -> None:
        """Register a canned structured dict response for prompts containing the substring."""
        self.canned_structured[prompt_substring] = response

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        self.call_history.append({
            "method": "complete",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "kwargs": kwargs,
        })

        if self.completion_handler:
            return self.completion_handler(prompt, system_prompt)

        for key, resp in self.canned_responses.items():
            if key in prompt:
                return resp

        return f"[FakeReasoningBackend response for: {prompt[:40]}...]"

    def structured_complete(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.call_history.append({
            "method": "structured_complete",
            "prompt": prompt,
            "schema": schema,
            "system_prompt": system_prompt,
            "kwargs": kwargs,
        })

        if self.structured_handler:
            return self.structured_handler(prompt, schema, system_prompt)

        for key, resp in self.canned_structured.items():
            if key in prompt:
                return resp

        # Default minimal valid structured response
        return {"result": "fake_structured_result", "prompt_snippet": prompt[:40]}

    def reset(self) -> None:
        """Clear canned responses and history."""
        self.canned_responses.clear()
        self.canned_structured.clear()
        self.call_history.clear()
        self.completion_handler = None
        self.structured_handler = None


def get_reasoning_backend(config: Optional[PersonBConfig] = None) -> ReasoningBackend:
    """Factory to instantiate configured reasoning backend."""
    cfg = config or default_config
    provider = cfg.reasoning_provider.lower().strip()

    if provider in ("fake", "mock", "test", "none", ""):
        return FakeReasoningBackend()

    # Provider adapters can be registered or extended here
    raise AdapterError(
        f"Unsupported or unconfigured reasoning provider: {provider}. "
        "Use 'fake' for offline testing or configure a custom provider adapter."
    )
