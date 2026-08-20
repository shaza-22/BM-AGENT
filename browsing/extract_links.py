"""
Link perception for the Banque Misr agent.

What it does
    Turns a fetched HTML page into the list of candidate next hops the planner
    LLM chooses from, and provides the URL normalisation that keeps the
    visited-set honest.

Inputs
    ``extract_links(raw_html, base_url)`` -- the raw HTML of a page and the URL
    it was fetched from (needed to resolve relative hrefs).
    ``normalize_url(url, base_url)`` -- a single href, absolute or relative.

Outputs
    ``extract_links`` -> ``[{"label": str, "url": str, "key": str,
    "source": "nav"|"body"|"footer", "is_pdf": bool}, ...]`` in document order,
    deduplicated by canonical key.
    ``normalize_url`` -> a fetchable absolute URL, or ``None`` if the link is
    unusable (wrong scheme, off-domain, a static asset, or the wrong language).
    ``canonical_key`` -> the identity string used for deduplication.

Why it is needed
    The agent has no index and no crawler: at every step, the only thing
    standing between it and a dead end is the quality of this link list. Two
    failure modes are fatal and both are handled here -- a link the LLM cannot
    see (missed by the selector, or labelled with an empty string), and a link
    that looks new but is not (the same page wearing a fresh session token,
    which drains the page budget on re-fetches).

Site-specific behaviour worth knowing about
    Every site fact this module encodes is structural (markup shape, URL
    grammar), never topical. There are no product names, no category keywords
    and no hardcoded paths anywhere in this layer.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Literal, TypedDict
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from bs4.element import Tag

from browsing import config

logger = logging.getLogger(__name__)

LinkSource = Literal["nav", "body", "footer"]


class LinkDict(TypedDict):
    """One candidate hop, as handed to the planner."""

    label: str
    url: str
    key: str
    source: LinkSource
    is_pdf: bool


# Region detection is a metadata best-effort. It must never drop a link: the
# fees hub is reachable only from the footer, so "footer == boilerplate" (the
# usual scraping assumption) would make a whole class of tasks unanswerable.
_FOOTER_RE = re.compile(r"footer|site-bottom|bottom-bar|copyright")
_NAV_RE = re.compile(r"(^|[^a-z])nav([^a-z]|$)|navbar|navigation|menu|header|topbar|masthead|breadcrumb")

_LABEL_ORIGIN_RANK = {"slug": 0, "attr": 1, "img-alt": 2, "heading": 3, "text": 4}


# --------------------------------------------------------------------------
# URL normalisation
# --------------------------------------------------------------------------
def _is_allowed_host(host: str) -> bool:
    """Exact domain or a real subdomain of it -- never a suffix match.

    ``host.endswith("banquemisr.com")`` also accepts ``evil-banquemisr.com``,
    which would let a single planted link walk the agent off-site and get the
    result cited as a Banque Misr source. ``digital.banquemisr.com`` is a
    legitimate subdomain and is allowed by the second branch.
    """
    if not host:
        return False
    domain = config.ALLOWED_DOMAIN.lower()
    return host == domain or host.endswith("." + domain)


def _extension_of(path: str) -> str:
    tail = path.rsplit("/", 1)[-1]
    dot = tail.rfind(".")
    return tail[dot:].lower() if dot > 0 else ""


def _detect_language(path: str, query: str) -> str | None:
    """Return the language tag a URL claims, or None if it is unmarked.

    Markers are inconsistent on this site: a ``sc_lang`` query parameter, an
    ``/ar-eg/`` path segment, or nothing at all. The query parameter wins
    because Sitecore lets it override the path.
    """
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key.lower() == config.LANG_PARAM:
            tag = value.strip().lower()
            if tag in config.KNOWN_LANGUAGE_TAGS:
                return tag
    for segment in path.split("/"):
        tag = unquote(segment).strip().lower()
        if tag in config.KNOWN_LANGUAGE_TAGS:
            return tag
    return None


def _normalize_path(path: str) -> str:
    """Decode, collapse, then re-encode so equivalent spellings converge.

    Spaces reach us three ways (literal, ``%20``, already-encoded) and Sitecore
    emits all three. Decoding first and re-quoting once gives one spelling.
    Path *case* is deliberately preserved: Sitecore paths can be case-sensitive
    and a lowercased URL may 404. Case-insensitive matching happens in
    ``canonical_key`` instead, which is used for identity, not for fetching.
    """
    decoded = unquote(path)
    decoded = re.sub(r"/{2,}", "/", decoded)
    quoted = quote(decoded, safe="/:@!$&'()*+,;=~-._")
    if len(quoted) > 1:
        quoted = quoted.rstrip("/") or "/"
    return quoted or "/"


def _filter_query(query: str) -> str:
    if not query:
        return ""
    kept = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key.lower() not in config.TRACKING_PARAMS
    ]
    kept.sort()
    return urlencode(kept)


def normalize_url(url: str, base_url: str) -> str | None:
    """Resolve, clean and vet a single href.

    Returns a fetchable absolute URL, or ``None`` when the link is not a usable
    hop: a non-HTTP scheme, off-domain, a static asset, or marked as a language
    other than :data:`config.LANGUAGE`.
    """
    if not url:
        return None
    raw = url.strip()
    # A bare "#" or "#section" is the same page; following it wastes a hop.
    if not raw or raw.startswith("#"):
        return None

    scheme_hint = urlsplit(raw).scheme.lower()
    if scheme_hint and scheme_hint in config.NON_FETCHABLE_SCHEMES:
        return None

    try:
        parts = urlsplit(urljoin(base_url, raw))
    except ValueError:
        logger.debug("unparseable href discarded: %r (base=%r)", url, base_url)
        return None

    if parts.scheme not in ("http", "https"):
        return None

    try:
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:  # malformed port
        return None
    if not _is_allowed_host(host):
        return None

    # Language is checked before parameters are stripped. sc_lang is itself a
    # stripped parameter, so doing this in the other order would erase the only
    # evidence that a URL is Arabic and let it through as English.
    if config.LANGUAGE:
        tag = _detect_language(parts.path, parts.query)
        if tag and tag.split("-")[0] != config.LANGUAGE.lower():
            return None

    path = _normalize_path(parts.path)
    if _extension_of(path) in config.ASSET_EXTENSIONS:
        return None

    netloc = host
    default_port = 80 if parts.scheme == "http" else 443
    if port and port != default_port:
        netloc = f"{host}:{port}"

    # The fragment is dropped: it never changes what the server returns.
    return urlunsplit((parts.scheme, netloc, path, _filter_query(parts.query), ""))


def canonical_key(url: str) -> str:
    """The identity of a page, for the visited-set and the per-run cache.

    Separate from :func:`normalize_url` on purpose. Normalisation must return
    something *fetchable*, which means preserving path case; identity must
    survive the site's inconsistent spelling of the same page
    (``Retail%20Banking`` vs ``retail-banking`` vs ``Retail_Banking``). Folding
    those together in the fetchable URL risks a 404; folding them here costs
    nothing and is what stops the agent re-fetching one page until its budget
    is gone.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = unquote(parts.path).lower()
    path = re.sub(r"[-_+]+", " ", path)
    path = re.sub(r"\s+", " ", path)
    path = re.sub(r"\s*/\s*", "/", path).strip().rstrip("/")
    query = parts.query.lower()
    return f"{host}{path}" + (f"?{query}" if query else "")


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _cap(label: str) -> str:
    """Cap on a word boundary so a heading+description link stays readable."""
    if len(label) <= config.MAX_LABEL_CHARS:
        return label
    clipped = label[: config.MAX_LABEL_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (clipped or label[: config.MAX_LABEL_CHARS]).rstrip() + "..."


def slug_label(url: str) -> str:
    """Derive a human label from the URL path.

    Unusually effective on this site: Sitecore paths read like prose
    (``/Pages/Credit%20Cards%20List`` -> "Credit Cards List"), so a link with
    no text at all still arrives at the planner describing itself.
    """
    path = unquote(urlsplit(url).path)
    segments = [s for s in path.split("/") if s]
    while segments:
        segment = segments[-1]
        stem, ext = segment, _extension_of(segment)
        if ext in config.STRIPPABLE_EXTENSIONS:
            stem = segment[: -len(ext)]
        noise = config.SLUG_NOISE_SEGMENTS
        if not stem.strip() or stem.lower() in noise or segment.lower() in noise:
            segments.pop()
            continue
        label = _clean_text(re.sub(r"[-_+]+", " ", stem))
        # Hyphenated slugs are lowercase ("apple-pay"); Sitecore titles already
        # carry their own capitalisation ("SMEs") that .title() would destroy.
        return label.title() if label and label == label.lower() else label
    return ""


def _first_attr(el: Tag, names: Iterable[str]) -> str:
    for name in names:
        value = el.get(name)
        if isinstance(value, str) and _clean_text(value):
            return _clean_text(value)
    for name in names:
        found = el.find(attrs={name: True})
        if isinstance(found, Tag):
            value = found.get(name)
            if isinstance(value, str) and _clean_text(value):
                return _clean_text(value)
    return ""


def _label_with_origin(el: Tag, url: str) -> tuple[str, str]:
    """The fallback chain, with the winning step reported for dedupe ranking.

    An unlabelled link is invisible to the LLM that picks the next hop, so the
    chain must not be able to return "". Homepage tiles are image-only and the
    main nav is Vue components, so the plain-text step misses often enough that
    the later steps carry real traffic.
    """
    text = _clean_text(el.get_text(" ", strip=True))
    if text:
        return _cap(text), "text"

    # Safety net for markup where the anchor's own text extraction comes back
    # empty but a descendant heading still holds a title.
    heading = el.find(["h1", "h2", "h3", "h4", "span"])
    if isinstance(heading, Tag):
        heading_text = _clean_text(heading.get_text(" ", strip=True))
        if heading_text:
            return _cap(heading_text), "heading"

    alt = _first_attr(el, ("alt",))
    if alt:
        return _cap(alt), "img-alt"

    attr = _first_attr(el, ("aria-label", "title"))
    if attr:
        return _cap(attr), "attr"

    return _cap(slug_label(url)) or url, "slug"


def link_label(el: Tag, url: str) -> str:
    """Best available human label for a link. Never returns an empty string."""
    return _label_with_origin(el, url)[0]


# --------------------------------------------------------------------------
# Region and type
# --------------------------------------------------------------------------
def link_source(el: Tag) -> LinkSource:
    """Tag the page region a link came from: metadata, never a filter.

    Footer wins over nav so that a ``<nav>`` nested inside a ``<footer>`` is
    still reported as footer. When detection is unreliable the answer is
    "body", because a mislabelled link is a debugging annoyance and a dropped
    link is an unanswerable task.
    """
    in_footer = in_nav = False
    for parent in el.parents:
        name = (parent.name or "").lower()
        if name == "footer":
            in_footer = True
        elif name in ("nav", "header"):
            in_nav = True

        classes = parent.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        haystack = " ".join(
            [*classes, str(parent.get("id") or ""), str(parent.get("role") or "")]
        ).lower()
        if haystack:
            if _FOOTER_RE.search(haystack):
                in_footer = True
            elif _NAV_RE.search(haystack) or parent.get("role") == "navigation":
                in_nav = True

    if in_footer:
        return "footer"
    if in_nav:
        return "nav"
    return "body"


def is_pdf_hint(url: str) -> bool:
    """Guess whether a URL is a downloadable document. A guess, not a verdict.

    Sitecore serves PDFs from ``/-/media/...ashx`` with no ``.pdf`` extension,
    so extension-sniffing alone misses them -- but that same prefix also serves
    images, so treating the prefix as proof would classify every slider graphic
    as a PDF. Known image extensions are therefore excluded here, and
    ``fetcher.fetch_page`` settles the question from the Content-Type header
    and the file's magic bytes.
    """
    path = unquote(urlsplit(url).path).lower()
    extension = _extension_of(path)
    if extension in config.DOCUMENT_EXTENSIONS:
        return True
    if config.MEDIA_PATH_MARKER in path and extension not in config.IMAGE_EXTENSIONS:
        return True
    return False


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def make_soup(raw_html: str) -> BeautifulSoup:
    """Parse with lxml when available, else the stdlib parser."""
    try:
        return BeautifulSoup(raw_html, "lxml")
    except Exception:  # pragma: no cover - depends on local install
        return BeautifulSoup(raw_html, "html.parser")


def extract_links(raw_html: str, base_url: str) -> list[LinkDict]:
    """Extract every followable link on a page, in document order."""
    if not raw_html:
        return []

    soup = make_soup(raw_html)
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    # Links whose target is also an <img src> on the same page are images
    # wearing an href (Sitecore's ".ashx" media). Dropping them here keeps the
    # planner from spending a page of its budget fetching a JPEG.
    image_keys = set()
    for img in soup.find_all("img", src=True):
        src = img.get("src")
        if isinstance(src, str):
            normalized = normalize_url(src, base_url)
            if normalized:
                image_keys.add(canonical_key(normalized))

    results: list[LinkDict] = []
    origins: dict[str, str] = {}
    positions: dict[str, int] = {}

    # Select on [href] rather than "a[href]". The main navigation is built from
    # Vue components -- <b-dropdown-item href="..."> and friends -- which carry
    # href on a non-anchor tag. soup.find_all("a", href=True) silently returns
    # zero main-nav links on this site, and the agent then cannot leave the
    # homepage. The footer does use real anchors, so both forms coexist.
    for el in soup.select("[href]"):
        if el.name in ("link", "base"):
            continue
        href = el.get("href")
        if not isinstance(href, str):
            continue

        url = normalize_url(href, base_url)
        if url is None:
            continue
        key = canonical_key(url)
        if key in image_keys:
            continue

        label, origin = _label_with_origin(el, url)

        if key in positions:
            # Keep the first occurrence's position and region (document order is
            # a weak relevance signal) but take the better label. Nav markup
            # comes first on every page and is often terser than the body copy
            # for the same destination.
            index = positions[key]
            if _LABEL_ORIGIN_RANK.get(origin, 0) > _LABEL_ORIGIN_RANK.get(origins[key], 0) or (
                origin == origins[key] and len(label) > len(results[index]["label"])
            ):
                results[index]["label"] = label
                origins[key] = origin
            continue

        positions[key] = len(results)
        origins[key] = origin
        results.append(
            LinkDict(
                label=label,
                url=url,
                key=key,
                source=link_source(el),
                is_pdf=is_pdf_hint(url),
            )
        )

    logger.debug("extracted %d links from %s", len(results), base_url)
    return results


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    # Runs against saved fixtures, never the live site, so this is safe to run
    # in a loop while iterating on the label chain.
    #
    #   python -m browsing.extract_links                       # synthetic fixtures
    #   python -m browsing.extract_links fixtures/live         # a directory
    #   python -m browsing.extract_links fixtures/live/index.html
    #   python -m browsing.extract_links "fixtures/live/*.html"
    import pathlib
    import sys

    from browsing._demo import FixtureNotFound, base_url_for, resolve_fixture_paths

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root = pathlib.Path(__file__).resolve().parent.parent
    try:
        files = resolve_fixture_paths(sys.argv[1:], root / "fixtures" / "synthetic")
    except FixtureNotFound as exc:
        print(exc)
        raise SystemExit(1) from None

    for path in files:
        base = base_url_for(path, "https://www.banquemisr.com/")
        links = extract_links(path.read_text(encoding="utf-8", errors="replace"), base)
        print(f"\n=== {path.name}: {len(links)} links (base={base}) ===")
        counts: dict[str, int] = {}
        for link in links:
            counts[link["source"]] = counts.get(link["source"], 0) + 1
            flag = " [pdf?]" if link["is_pdf"] else ""
            print(f"  {link['source']:<6} {link['label'][:60]:<60} {link['url']}{flag}")
        print(f"  regions: {counts}")
