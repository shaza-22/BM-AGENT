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

    # fixtures/live/ has no Arabic snapshots yet, so each language seed that is
    # not already routed falls back to the English homepage. The page content
    # is not what these tests check -- whether a run can fetch its own starting
    # page is. Once save_fixtures.py has pulled the Arabic pages this resolves
    # to the real one automatically.
    from agent.config import seed_for
    from browsing import config as browsing_config

    english_home = routes.get("https://www.banquemisr.com/")
    if english_home is not None:
        for language in browsing_config.LANGUAGE_URL_TAGS:
            seed = seed_for(language)
            if seed not in routes:
                routes[seed] = FakeResponse(
                    seed, content=english_home.content, content_type="text/html"
                )
    return routes


def live_fetcher(**kwargs):
    """A real Fetcher wired to the saved pages, with no delay and no network."""
    from browsing.fetcher import Fetcher

    kwargs.setdefault("respect_robots", False)
    kwargs.setdefault("delay_range", (0.0, 0.0))
    kwargs.setdefault("allow_playwright", False)
    return Fetcher(session=FakeSession(live_routes()), **kwargs)


PLANNING_MARKER = "Break this into the sub-goals needed to answer it"
# Imported rather than duplicated: the composer has two prompt variants and a
# marker copied here would silently stop matching when one of them changes.
from agent.answer import COMPOSING_MARKER  # noqa: E402  (re-exported for tests)
EXTRACTION_MARKER = "Pull out the facts on this page"


def no_facts_reply() -> str:
    """An extraction reply saying the page does not answer the question.

    The honest default for a stub: the fallback fires only where the keyword
    path failed, and "this page does not answer it" is often the true answer.
    """
    return json.dumps({"facts": [], "present": False, "note": "stubbed: no facts"})


def copy_from_page(prompt: str, count: int = 2, plus_invented: bool = False) -> str:
    """Extraction reply copying real lines out of the page in the prompt.

    This is what a well-behaved model does -- values character for character
    from the page it was shown -- so a test using it exercises the verbatim
    check honestly rather than against text the run never fetched.
    """
    body = prompt.split("---", 1)[1].rsplit("---", 1)[0]
    lines = [l.strip() for l in body.splitlines() if len(l.strip()) > 12][:count]
    facts = [{"label": f"fact {i + 1}", "value": line} for i, line in enumerate(lines)]
    if plus_invented:
        facts.append({"label": "invented", "value": "EGP 99999 nowhere on this page"})
    return json.dumps({"facts": facts, "present": True, "note": ""})


def plan_reply(*questions: str, reasoning: str = "decomposed for the test") -> str:
    """A planner reply with the given sub-goals, as agent/planner.py expects."""
    return json.dumps({
        "sub_goals": [{"question": q, "why": "needed for the test"} for q in questions],
        "reasoning": reasoning,
    })


def echo_plan(prompt: str) -> str:
    """A one-sub-goal plan restating the task, taken from the prompt itself.

    The planning call happens before any navigation, so a script that only
    knows about link choices would have its first choice eaten by it. This
    answers planning faithfully -- one sub-goal, which is what the keyword
    planner produced for nearly every task -- so tests written before planning
    existed keep testing what they were written to test.
    """
    task = ""
    marker = "A user asked a bank's research assistant:"
    if marker in prompt:
        task = prompt.split(marker, 1)[1].strip().splitlines()[0].strip()
    return plan_reply(task or "Find information for the request")


def choose_by(*needles: str):
    """A FakeLLMClient callable that picks the first candidate line matching.

    Tests drive navigation by *label*, not by index: ranking decides the
    numbering, so a hardcoded index would break whenever a weight changes and
    would tell us nothing about the behaviour under test.

    Planning and composition prompts are answered separately and do not consume
    a needle -- the script is about where the agent walks.
    """
    remaining = list(needles)

    def respond(prompt: str) -> str:
        if PLANNING_MARKER in prompt:
            return echo_plan(prompt)
        if EXTRACTION_MARKER in prompt:
            return no_facts_reply()
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
def api_error(message: str, *, code: int | None = None, details: object = None) -> Exception:
    """An SDK-shaped error: a status code and a google.rpc details payload.

    Real refusals carry a status; the message often says nothing useful. Tests
    that fake a bare ValueError would let status-driven handling pass by
    accident.
    """
    error = RuntimeError(message)
    if code is not None:
        error.code = code
    if details is not None:
        error.details = details
    return error


def quota_error(quota_id: str, *, retry_delay: str | None = None) -> Exception:
    """A 429 shaped like Google's, with a QuotaFailure violation."""
    details: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_requests",
                    "quotaId": quota_id,
                }
            ],
        }
    ]
    if retry_delay:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    return api_error(
        "429 RESOURCE_EXHAUSTED. You exceeded your current quota.",
        code=429,
        details={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}},
    )


def bad_request(message: str = "Request contains an invalid argument.") -> Exception:
    """Google's bare 400: no field named, so only the status is usable."""
    return api_error(f"400 INVALID_ARGUMENT. {message}", code=400)


def grow_pdf(content: bytes, copies: int) -> bytes:
    """Repeat a PDF's pages, to build a document large enough to change path."""
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for _ in range(copies):
        for page in PdfReader(io.BytesIO(content)).pages:
            writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def no_sleep(_seconds: float) -> None:
    """Substituted for time.sleep so retry tests never actually wait."""


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

    def __init__(self, response=None, beta_error: Exception | None = None,
                 errors_until: int | None = None):
        self.response = response or text_response("ok")
        self.beta_error = beta_error
        # None raises on every call; N raises on the first N only, which is how
        # a transient failure that later recovers is simulated.
        self.errors_until = errors_until
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
                if outer.beta_error and (
                    outer.errors_until is None or len(outer.beta_calls) <= outer.errors_until
                ):
                    raise outer.beta_error
                return outer.response

        self.messages = _Messages()
        self.beta = types.SimpleNamespace(messages=_BetaMessages())



class StubGemini:
    """Minimal stand-in for google.genai.Client."""

    def __init__(self, text: str = "ok", error: Exception | None = None,
                 candidates=None, errors_until: int | None = None):
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
                if outer.error and (
                    outer.errors_until is None or len(outer.configs) <= outer.errors_until
                ):
                    raise outer.error
                return types.SimpleNamespace(
                    text=outer.text, candidates=outer.candidates, prompt_feedback=None
                )

        self.models = _Models()


