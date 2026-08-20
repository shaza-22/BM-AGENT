"""
Snapshot live Banque Misr pages to ``fixtures/live/`` for offline testing.

What it does
    Fetches every URL listed in ``fixtures/fixture_urls.txt``, saves the raw
    HTML and the cleaned text, then discovers one linked PDF from whatever was
    fetched and saves that too. Writes a ``manifest.json`` describing every
    attempt, including the failures.

Inputs
    ``fixtures/fixture_urls.txt`` -- one URL or path per line.
    ``--out``, ``--limit``, ``--no-pdf``, ``--allow-playwright`` (see --help).

Outputs
    ``fixtures/live/<slug>.html``      raw HTML, exactly as served
    ``fixtures/live/<slug>.txt``       cleaned text from html_to_text
    ``fixtures/live/<slug>.pdf``       raw bytes of the discovered PDF
    ``fixtures/live/<slug>.pdf.txt``   its extracted text
    ``fixtures/live/manifest.json``    per-URL status, canonical key, timing

Why it is needed
    Run once, then iterate on parsing offline. Nothing else in the project
    should touch the live site during development: the pages are slow, the
    session tokens change on every request, and hammering a bank's website to
    debug a regex is not acceptable.

Usage
    python scripts/save_fixtures.py

Note
    The listed paths were traced by hand and may be stale. A URL that 404s is
    recorded in the manifest and the run continues -- one bad path must not
    cost the whole snapshot.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
import time
from urllib.parse import urljoin, urlsplit

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from browsing import config  # noqa: E402
from browsing.extract_links import canonical_key, extract_links, normalize_url  # noqa: E402
from browsing.fetcher import Fetcher, RateLimiter, pdf_to_text  # noqa: E402

logger = logging.getLogger("save_fixtures")

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL_LIST = ROOT / "fixtures" / "fixture_urls.txt"
DEFAULT_OUT = ROOT / "fixtures" / "live"
SEED = f"https://www.{config.ALLOWED_DOMAIN}/"
MAX_PDF_ATTEMPTS = 4


def read_url_list(path: pathlib.Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing URL list: {path}")
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            urls.append(urljoin(SEED, entry))
    return urls


def slugify(url: str) -> str:
    """Filesystem-safe name derived from the canonical key, so two spellings of
    one page cannot produce two fixture files."""
    key = canonical_key(url).removeprefix(config.ALLOWED_DOMAIN)
    slug = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
    return (slug or "index")[:100]


def save_page(page: dict, out_dir: pathlib.Path) -> dict:
    record = {
        "requested_url": page["requested_url"],
        "url": page["url"],
        "key": canonical_key(page["url"]),
        "status": page["status"],
        "ok": page["ok"],
        "content_type": page["content_type"],
        "render_mode": page["render_mode"],
        "error": page["error"],
        "fetched_at": page["fetched_at"],
        "files": [],
    }
    if not page["ok"]:
        return record

    slug = slugify(page["url"])
    if page["raw_html"]:
        html_path = out_dir / f"{slug}.html"
        html_path.write_text(page["raw_html"], encoding="utf-8")
        record["files"].append(html_path.name)
        record["html_bytes"] = len(page["raw_html"])
    if page["text"]:
        text_path = out_dir / f"{slug}.txt"
        text_path.write_text(page["text"], encoding="utf-8")
        record["files"].append(text_path.name)
        record["text_chars"] = len(page["text"])
    if page["raw_html"]:
        record["link_count"] = len(extract_links(page["raw_html"], page["url"]))
    return record


def discover_pdf(pages: list[dict]) -> list[str]:
    """Candidate document URLs, found by following links rather than guessing.

    Nothing here knows which page hosts the fee documents; every fetched page
    contributes its own PDF-hinted links, ordered so that the ones that look
    most like documents come first.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for page in pages:
        if not page.get("raw_html"):
            continue
        for link in extract_links(page["raw_html"], page["url"]):
            if link["is_pdf"] and link["key"] not in seen:
                seen.add(link["key"])
                candidates.append(link["url"])
    candidates.sort(key=lambda url: 0 if url.lower().endswith(".pdf") else 1)
    return candidates


def save_pdf(
    url: str, fetcher: Fetcher, session: requests.Session, limiter: RateLimiter,
    out_dir: pathlib.Path,
) -> dict | None:
    """Fetch one candidate document and keep it only if it really is a PDF.

    Done with a direct GET rather than ``Fetcher.fetch`` so the raw bytes can be
    saved alongside the extracted text -- the teammate's extraction code needs
    the real file, not just my rendering of it. robots.txt and the politeness
    delay are still applied.
    """
    if not fetcher.robots_allowed(url):
        logger.warning("robots.txt disallows candidate %s", url)
        return None

    limiter.wait(urlsplit(url).hostname or "")
    try:
        response = session.get(url, timeout=config.REQUEST_TIMEOUT_S, allow_redirects=True)
    except requests.RequestException as exc:
        logger.warning("candidate %s failed: %s", url, exc)
        return None

    header = response.headers.get("Content-Type", "")
    content = response.content
    is_pdf = content[:5] == b"%PDF-" or "application/pdf" in header.lower()
    if response.status_code >= 400 or not is_pdf:
        logger.info(
            "candidate %s is not a PDF (HTTP %s, %s) -- trying the next one",
            url, response.status_code, header or "no content-type",
        )
        return None

    slug = slugify(str(response.url))
    pdf_path = out_dir / f"{slug}.pdf"
    pdf_path.write_bytes(content)
    text, error = pdf_to_text(content)
    record = {
        "requested_url": url,
        "url": normalize_url(str(response.url), url) or url,
        "key": canonical_key(url),
        "status": response.status_code,
        "ok": text is not None,
        "content_type": "pdf",
        "render_mode": "requests",
        "error": error,
        "files": [pdf_path.name],
        "pdf_bytes": len(content),
    }
    if text:
        text_path = out_dir / f"{slug}.pdf.txt"
        text_path.write_text(text, encoding="utf-8")
        record["files"].append(text_path.name)
        record["text_chars"] = len(text)
    logger.info("saved PDF %s (%d bytes)", pdf_path.name, len(content))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--urls", type=pathlib.Path, default=URL_LIST)
    parser.add_argument("--limit", type=int, default=0, help="stop after N URLs")
    parser.add_argument("--no-pdf", action="store_true", help="skip PDF discovery")
    parser.add_argument(
        "--allow-playwright", action="store_true",
        help="allow browser escalation for under-rendered pages",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    urls = read_url_list(args.urls)
    if args.limit:
        urls = urls[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    limiter = RateLimiter()
    fetcher = Fetcher(
        session=session, rate_limiter=limiter, allow_playwright=args.allow_playwright
    )

    started = time.perf_counter()
    records: list[dict] = []
    pages: list[dict] = []

    for url in urls:
        page = fetcher.fetch(url)
        pages.append(page)
        record = save_page(page, args.out)
        records.append(record)
        if not page["ok"]:
            # A hand-traced path may simply be stale; note it and keep going.
            logger.warning("SKIPPED %s -- HTTP %s (%s)", url, page["status"], page["error"])

    if not args.no_pdf:
        candidates = discover_pdf(pages)
        logger.info("discovered %d PDF-hinted links", len(candidates))
        for candidate in candidates[:MAX_PDF_ATTEMPTS]:
            record = save_pdf(candidate, fetcher, session, limiter, args.out)
            if not record:
                continue
            records.append(record)
            if record["ok"]:
                break
            # A real PDF that yields no text is image-only (a scanned guide).
            # It is saved, but it is useless as extraction test data, so keep
            # trying candidates until one produces readable text.
            logger.info("saved %s but it has no extractable text; trying the next candidate",
                        record["requested_url"])
        else:
            logger.warning("no candidate link produced a PDF with extractable text")

    manifest = {
        "generated_at": records[0]["fetched_at"] if records else None,
        "seed": SEED,
        "language": config.LANGUAGE,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "stats": fetcher.stats,
        "pages": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    saved = sum(1 for record in records if record["ok"])
    failed = [record for record in records if not record["ok"]]
    print(f"\nsaved {saved}/{len(records)} into {args.out}")
    print(f"fetcher stats: {fetcher.stats}")
    for record in failed:
        print(f"  FAILED {record['requested_url']} -> HTTP {record['status']} {record['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
