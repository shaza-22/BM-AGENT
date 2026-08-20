"""
Watch the agent navigate, one sub-goal at a time.

What it does
    Runs the navigation loop for a sub-goal given on the command line and
    prints the trail as it happens: which link was chosen at each hop, why, and
    what the page turned out to be.

Inputs
    A sub-goal as the positional argument, plus:
      --offline            replay fixtures/live/ instead of fetching the site
      --resolve-on WORD    stand-in validator: resolve when the page text
                           contains WORD (the real validator is a separate
                           deliverable; this only lets a demo run terminate)
      --provider / --model / --effort / --max-hops / --max-pages / --log FILE

Outputs
    A human-readable trail on stdout, and optionally the JSON Lines step log.

Why it is needed
    Not run by pytest, and deliberately so: this is the demo and the
    confidence check. --offline is the one to reach for while tuning the
    prompt -- it exercises the real model against the real saved pages without
    touching the live site, which matters because the site is behind a WAF
    that bans on burst traffic.

Usage
    python scripts/live_navigate.py "Find the annual fee of the classic credit card"
    python scripts/live_navigate.py "Where are the fee schedules?" --offline
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from agent.llm import make_llm_client  # noqa: E402
from agent.navigator import Navigator  # noqa: E402
from agent.trail_log import StepLogger  # noqa: E402
from browsing.fetcher import Fetcher  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIVE = ROOT / "fixtures" / "live"


class _FixtureResponse:
    """Enough of requests.Response for the Fetcher to replay a saved page."""

    def __init__(self, url: str, content: bytes, content_type: str) -> None:
        self.url, self.content = url, content
        self.status_code = 200 if content else 404
        self.headers = {"Content-Type": content_type}

    def iter_content(self, chunk_size: int = 65536):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def close(self) -> None:
        pass


class _FixtureSession:
    """Serves fixtures/live/ by URL, so --offline never touches the network."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.routes: dict[str, tuple[bytes, str]] = {}
        manifest = LIVE / "manifest.json"
        if not manifest.is_file():
            raise SystemExit(f"no fixtures in {LIVE} -- run scripts/save_fixtures.py first")
        for record in json.loads(manifest.read_text(encoding="utf-8")).get("pages", []):
            url = record.get("url")
            for name in record.get("files") or []:
                path = LIVE / name
                if not (url and path.is_file()):
                    continue
                if name.endswith(".html"):
                    self.routes[url] = (path.read_bytes(), "text/html")
                elif name.endswith(".pdf"):
                    self.routes[url] = (path.read_bytes(), "application/pdf")
        self.routes.setdefault(config.SEED_URL, self.routes.get(config.SEED_URL, (b"", "text/html")))

    def get(self, url: str, **kwargs):
        content, content_type = self.routes.get(url, (b"", "text/html"))
        return _FixtureResponse(url, content, content_type)


def make_contains_validator(word: str):
    """A stand-in for the real validator, so a demo run can actually finish.

    Deliberately crude and driven entirely by the command line -- it holds no
    knowledge of any product or category, and it is not what the finished agent
    will use to decide that a sub-goal is answered.
    """
    needle = word.lower()

    def validate(sub_goal: str, page: dict) -> dict:
        hit = needle in (page.get("text") or "").lower()
        return {
            "resolved": hit,
            "extracted": {"url": page["url"], "chars": len(page.get("text") or "")} if hit else {},
            "reason": f"page {'contains' if hit else 'does not contain'} {word!r}",
        }

    return validate


def build_fetcher(offline: bool) -> Fetcher:
    if offline:
        return Fetcher(
            session=_FixtureSession(), respect_robots=False,
            delay_range=(0.0, 0.0), allow_playwright=False,
        )
    # Live: keep the politeness delay and robots handling exactly as shipped.
    return Fetcher()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("sub_goal")
    parser.add_argument("--offline", action="store_true", help="replay fixtures/live/")
    parser.add_argument("--resolve-on", metavar="WORD", help="stand-in validator")
    parser.add_argument("--max-hops", type=int, default=config.MAX_HOPS)
    parser.add_argument("--max-pages", type=int, default=config.MAX_PAGES)
    parser.add_argument("--provider", default=config.PROVIDER, choices=["gemini", "claude"])
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--effort", help="Claude only: low | medium | high | xhigh | max")
    parser.add_argument("--log", type=pathlib.Path, help="write the JSON Lines step log here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    validate_fn = make_contains_validator(args.resolve_on) if args.resolve_on else None

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.effort:
        if args.provider != "claude":
            parser.error("--effort applies to the Claude provider only")
        overrides["effort"] = args.effort

    stream = args.log.open("w", encoding="utf-8") if args.log else None
    navigator = Navigator(
        make_llm_client(args.provider, **overrides),
        fetcher=build_fetcher(args.offline),
        validate_fn=validate_fn,
        step_logger=StepLogger(stream),
        max_hops=args.max_hops,
        max_pages=args.max_pages,
    )

    print(f"sub-goal : {args.sub_goal}")
    print(f"seed     : {config.SEED_URL}")
    print(f"mode     : {'offline (saved pages)' if args.offline else 'LIVE SITE'}")
    print(f"provider : {args.provider}")
    print(f"caps     : {args.max_hops} hops / {args.max_pages} pages")
    print("-" * 78)

    started = time.perf_counter()
    try:
        result = navigator.navigate(args.sub_goal)
    finally:
        if stream:
            stream.close()

    for step in result.trail:
        head = f"hop {step.hop}"
        if step.label:
            origin = "" if step.discovered_at_hop is None else f" (link seen at hop {step.discovered_at_hop})"
            print(f"{head}: followed {step.label!r} [{step.source}]{origin}")
        else:
            print(f"{head}: seed page")
        print(f"       url    {step.url}")
        print(f"       why    {step.reasoning}")
        print(
            f"       page   status={step.fetch_status} type={step.content_type} "
            f"links={step.links_found} offered={step.candidates_offered}/{step.candidates_available} "
            f"confidence={step.confidence:.2f} {step.elapsed_ms}ms"
        )
        print(f"       verdict {'RESOLVED' if step.validated else 'not resolved'} -- {step.validate_reason}")
        print()

    print("-" * 78)
    print(f"status   : {result.status}" + (f" (cap: {result.cap_hit})" if result.cap_hit else ""))
    print(f"reason   : {result.final_reasoning}")
    print(f"hops     : {result.hops_used} | pages: {result.pages_fetched} | "
          f"{time.perf_counter() - started:.1f}s")
    print(f"llm calls: {result.stats.get('llm_calls')}")
    if result.extracted:
        print(f"extracted: {result.extracted}")
    print("sources  :")
    for url in result.sources:
        print(f"  - {url}")
    if args.log:
        print(f"\nstep log written to {args.log}")

    return 0 if result.status in ("resolved", "no_candidates") else 1


if __name__ == "__main__":
    raise SystemExit(main())
