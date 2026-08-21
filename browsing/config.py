"""
Configuration for the Banque Misr agent's browsing (perception) layer.

What it does
    Holds every tunable used by ``fetcher.py`` and ``extract_links.py`` in one
    place: the language policy, the domain allow-list, politeness settings,
    and the thresholds that decide when a page looks under-rendered.

Inputs
    None. This module is pure data and imports nothing from the project.

Outputs
    Module-level constants.

Why it is needed
    Two reasons. First, the agent must not contain any product-category
    knowledge -- no URLs, no topic keywords, no per-page branching -- so the
    only site-specific facts allowed anywhere in this layer are the generic
    ones collected here (the domain, Sitecore's parameter names, the language
    markers). Second, behaviour that the project is graded on (how often we
    escalate to a browser, how polite we are) has to be adjustable without
    editing logic.

Note on read-only-ness
    These are constants, not state. Nothing in this layer mutates them at
    runtime; per-run state lives on a ``Fetcher`` instance instead, so tasks
    can run concurrently without sharing anything.
"""

from __future__ import annotations

# --- Language policy -------------------------------------------------------
# The site is bilingual with inconsistent markers: sometimes "?sc_lang=ar-EG",
# sometimes an "/ar-eg/" path segment, often no marker at all. Without a filter
# the agent follows an Arabic link mid-task and the rest of the run is in a
# language the planner did not ask for. Set to None to disable the filter.
LANGUAGE: str | None = "en"

# Language markers are only recognised from this set. A bare regex like
# ^[a-z]{2}(-[a-z]{2})?$ would misread a real content segment (an "sm" or "ar"
# folder) as a language tag and silently drop reachable pages.
KNOWN_LANGUAGE_TAGS: frozenset[str] = frozenset(
    {"en", "ar", "fr", "en-us", "en-gb", "en-eg", "ar-eg", "fr-fr"}
)

# Sitecore's language query parameter. Checked *before* parameter stripping --
# strip it first and an Arabic URL becomes indistinguishable from an English one.
LANG_PARAM = "sc_lang"

# --- Domain policy ---------------------------------------------------------
# Matched as (host == ALLOWED_DOMAIN or host.endswith("." + ALLOWED_DOMAIN)).
# A plain endswith() would also accept "evil-banquemisr.com", which is an
# off-site hop the agent would happily follow and then cite as a source.
ALLOWED_DOMAIN = "banquemisr.com"

# --- HTTP politeness -------------------------------------------------------
USER_AGENT = (
    "BanqueMisrResearchAgent/0.1 (academic research assistant project; "
    "respects robots.txt; contact: project maintainer)"
)
REQUEST_TIMEOUT_S = 20.0
DELAY_RANGE_S: tuple[float, float] = (1.0, 2.0)  # per-host, randomised
MAX_RETRIES = 1
RETRY_BACKOFF_S = 2.0

# Transient -> worth one retry.
RETRY_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})
# Definitive -> retrying burns budget for nothing. 403 is usually a bot wall,
# which a retry never fixes; it is an escalation candidate instead.
NO_RETRY_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 405, 410})

# Refuse to buffer anything absurd (a video, a disk image) into memory.
MAX_CONTENT_BYTES = 25 * 1024 * 1024

# --- PDF extraction --------------------------------------------------------
# pdfplumber keeps column structure, which is why it is preferred: a fee table
# flattened into prose is unreadable. But its per-page layout analysis is
# expensive and the cost grows with the document.
#
# Measured on an 84-page, 6.9MB tariff (129k characters of text):
#   pdfplumber, text + tables   24.4s      full structure
#   pdfplumber, text only       15.3s      no table rendering
#   pypdf, text only             6.3s      all the content, flat
# On a 7-page document the same comparison is 1.8s / 1.2s / 2.2s -- pypdf is
# actually slower there. So the right extractor depends on size, and above this
# page count the run switches to the flat-text path: no content is lost, only
# column alignment, and a document this large is past the point where the text
# would be handed to a model whole anyway.
PDF_FIDELITY_MAX_PAGES = 25

# Table rendering is 37% of pdfplumber's cost. Turning it off keeps the prose
# and loses the column structure that makes a fee row readable.
PDF_EXTRACT_TABLES = True

# --- Render-escalation thresholds -----------------------------------------
# From an earlier survey, ~1931 pages parsed fine with plain requests and only
# ~65 needed a headless browser. Escalating unconditionally would pay browser
# cost on ~97% of fetches, so we escalate only on evidence of under-rendering.
ESCALATE_MIN_LINKS = 5
ESCALATE_MIN_TEXT_CHARS = 200
# Signal D fires only when a shell marker is present AND text is still well
# under a comfortable floor, so a genuinely short-but-real page is left alone.
ESCALATE_SHELL_TEXT_MULTIPLIER = 4
SPA_SHELL_IDS: frozenset[str] = frozenset({"app", "root", "__nuxt", "__next", "vue-app"})
PLAYWRIGHT_NETWORKIDLE_MS = 5000

# --- URL normalisation -----------------------------------------------------
# "csrt" is a per-request anti-CSRF token: the same page yields a different one
# every time. Left in place, the visited-set never dedupes and the agent
# re-fetches one page until its page budget is gone. The sc_* parameters are
# Sitecore rendering hints that likewise do not change the content identity.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "csrt",
        "sc_site",
        "sc_mode",
        "sc_itemid",
        "sc_lang",
        "sc_version",
        "sc_database",
        "sc_device",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
    }
)

NON_FETCHABLE_SCHEMES: frozenset[str] = frozenset(
    {"mailto", "tel", "javascript", "data", "file", "sms", "fax", "callto", "whatsapp"}
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp", ".bmp", ".avif"}
)
# Static assets are never an answer and never a useful hop.
ASSET_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | frozenset(
    {".css", ".js", ".mjs", ".map", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".webm", ".zip"}
)

DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".doc", ".docx", ".xls", ".xlsx"})

# Sitecore serves media (PDFs *and* images) from this prefix, usually with a
# ".ashx" extension rather than a real one. So the path is a hint that a link
# may be a document -- never proof. Content-Type decides; see fetcher.py.
MEDIA_PATH_MARKER = "/-/media/"

# Structural path segments that carry no meaning in a slug-derived label.
# Checked against both the raw segment and its extension-stripped stem, so
# "default", "default.aspx" and "Default.ASPX" all resolve to noise.
SLUG_NOISE_SEGMENTS: frozenset[str] = frozenset(
    {"home", "pages", "page", "sitecore", "content", "default", "index", "aspx"}
)
STRIPPABLE_EXTENSIONS: frozenset[str] = frozenset({".aspx", ".ashx", ".html", ".htm", ".php"})

# --- Invisible characters --------------------------------------------------
# Sitecore emits zero-width spaces inside labels ("Get a Card\u200b Use BM
# cards..."). Python's \s does NOT match U+200B or U+FEFF, so they survive
# whitespace collapsing, inflate label length and waste prompt context.
# U+00A0 is matched by \s but is normalised here so it never reaches a label
# as a non-breaking space.
INVISIBLE_TRANSLATION = {
    0x200B: None,  # zero width space
    0x200C: None,  # zero width non-joiner
    0x200D: None,  # zero width joiner
    0x200E: None,  # left-to-right mark
    0x200F: None,  # right-to-left mark
    0xFEFF: None,  # byte order mark / zero width no-break space
    0x00AD: None,  # soft hyphen
    0x00A0: " ",   # non-breaking space
}

# --- Alternate URL schemes -------------------------------------------------
# The same content is served under two path schemes:
#   /en/ABOUT-US/History   (appears in nav)
#   /Home/ABOUT%20US/History   (appears in body/footer)
# canonical_key already folds hyphens to spaces, so after keying these differ
# only in the leading segment. Stripping a leading segment from this set gives
# an alias key that links the two. This is a HEURISTIC -- the two forms are not
# guaranteed to be identical pages -- so it is only ever used to deprioritise
# and warn, never to drop a link or to key the cache.
URL_SCHEME_PREFIXES: frozenset[str] = frozenset({"en", "home"})

# --- Labels ----------------------------------------------------------------
# Some links wrap a heading *and* a description; the combined text is useful
# context for the planner but must not blow up the prompt.
MAX_LABEL_CHARS = 180
