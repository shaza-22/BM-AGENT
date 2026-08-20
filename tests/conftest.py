"""Shared offline test doubles.

Nothing in the test suite touches the network: HTTP is served by ``FakeSession``
and HTML comes from inline strings or ``fixtures/synthetic/``.
"""

from __future__ import annotations

import pathlib
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
