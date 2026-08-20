"""Shared offline test doubles.

Nothing in the test suite touches the network: HTTP is served by ``FakeSession``
and HTML comes from inline strings or ``fixtures/synthetic/``.
"""

from __future__ import annotations

import json
import pathlib
import types
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "synthetic"


class FakeResponse:
    """Minimal stand-in for ``requests.Response`` (streamed reads included)."""

    def __init__(
        self,
        url: str,
        status_code: int = 200,
        content: bytes = b"",
        content_type: str = "text/html",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type, **(extra_headers or {})}
        self.closed = False

    def iter_content(self, chunk_size: int = 65536):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Routes URLs to responses (or exceptions) and records every call."""

    def __init__(self, routes: dict[str, object] | None = None) -> None:
        self.headers: dict[str, str] = {}
        self.routes = dict(routes or {})
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        route = self.routes.get(url, self.routes.get("*"))
        if route is None:
            return FakeResponse(url, status_code=404, content=b"not found")
        if isinstance(route, list):  # successive responses for retry tests
            route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, BaseException):
            raise route
        if callable(route):
            return route(url)
        return route


def html_response(url: str, body: str, **kwargs) -> FakeResponse:
    return FakeResponse(url, content=body.encode("utf-8"), **kwargs)


@pytest.fixture
def fixture_html():
    def _load(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return _load


def minimal_pdf(lines: list[str]) -> bytes:
    """Build a tiny but genuinely valid PDF, so the PDF path is really exercised."""
    content = "BT /F1 12 Tf 72 720 Td 14 TL\n" + "".join(f"({line}) Tj T*\n" for line in lines) + "ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = "%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n{obj}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n"
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


# --------------------------------------------------------------------------
# Live-fixture support for the navigation tests
# --------------------------------------------------------------------------
LIVE = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "live"


def live_manifest() -> dict:
    manifest = LIVE / "manifest.json"
    if not manifest.is_file():
        return {"pages": []}
    return json.loads(manifest.read_text(encoding="utf-8"))


def live_routes() -> dict[str, object]:
    """Map each saved page's real URL to a response replaying it.

    Serving the fixtures through a FakeSession means the tests exercise the
    real Fetcher -- decoding, classification, caching, escalation -- rather
    than a stub that would hide a regression in any of them.
    """
    from browsing.extract_links import canonical_key

    routes: dict[str, object] = {}
    for record in live_manifest().get("pages", []):
        url = record.get("url")
        if not url:
            continue
        for name in record.get("files") or []:
            path = LIVE / name
            if not path.is_file():
                continue
            if name.endswith(".html"):
                routes[url] = FakeResponse(url, content=path.read_bytes(), content_type="text/html")
            elif name.endswith(".pdf"):
                routes[url] = FakeResponse(
                    url, content=path.read_bytes(), content_type="application/pdf"
                )
    # The seed is saved under its canonical form; accept the bare host too.
    for url in list(routes):
        if canonical_key(url) == "banquemisr.com":
            routes.setdefault("https://www.banquemisr.com/", routes[url])
    return routes


def live_fetcher(**kwargs):
    """A real Fetcher wired to the saved pages, with no delay and no network."""
    from browsing.fetcher import Fetcher

    kwargs.setdefault("respect_robots", False)
    kwargs.setdefault("delay_range", (0.0, 0.0))
    kwargs.setdefault("allow_playwright", False)
    return Fetcher(session=FakeSession(live_routes()), **kwargs)


def choose_by(*needles: str):
    """A FakeLLMClient callable that picks the first candidate line matching.

    Tests drive navigation by *label*, not by index: ranking decides the
    numbering, so a hardcoded index would break whenever a weight changes and
    would tell us nothing about the behaviour under test.
    """
    remaining = list(needles)

    def respond(prompt: str) -> str:
        needle = remaining.pop(0) if remaining else None
        if needle is None:
            return json.dumps({"choice": -1, "reasoning": "script exhausted", "confidence": 0.4})
        for line in prompt.splitlines():
            index, sep, rest = line.partition(" | ")
            if sep and index.strip().isdigit() and needle.lower() in rest.lower():
                return json.dumps(
                    {
                        "choice": int(index.strip()),
                        "reasoning": f"Following {needle!r} because it should lead to the answer.",
                        "confidence": 0.8,
                    }
                )
        return json.dumps(
            {"choice": -1, "reasoning": f"no candidate matched {needle!r}", "confidence": 0.3}
        )

    return respond


# --------------------------------------------------------------------------
# LLM provider stubs
# --------------------------------------------------------------------------
# A syntactically valid Google API key that is not one. Used to prove a key
# never reaches an error message, a log line or a repr.
FAKE_KEY = "AIzaSyFAKE0000000000000000000000000000"


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



class StubGemini:
    """Minimal stand-in for google.genai.Client."""

    def __init__(self, text: str = "ok", error: Exception | None = None,
                 candidates=None, errors_until: int = 0):
        self.text, self.error = text, error
        self.candidates = candidates
        self.errors_until = errors_until
        self.configs: list[object] = []
        outer = self

        class _Models:
            def generate_content(self, *, model, contents, config):
                outer.configs.append(config)
                outer.last_model = model
                outer.last_contents = contents
                if outer.error and len(outer.configs) <= max(outer.errors_until, 1):
                    raise outer.error
                return types.SimpleNamespace(
                    text=outer.text, candidates=outer.candidates, prompt_feedback=None
                )

        self.models = _Models()


