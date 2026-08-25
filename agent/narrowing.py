"""
Deciding when an overview page is a signpost rather than an answer.

What it does
    ``find_narrowing_link(sub_goal, current_url, links, visited)`` answers one
    question: *does this page point at a page that is more specifically about
    what was asked?* It returns that link, or ``None``.

Inputs
    The sub-goal text, the URL of the page the agent is standing on, the links
    that page offers, and the set of canonical keys already visited this run.

Outputs
    A link dict from :mod:`browsing.extract_links`, or ``None`` when the page
    offers nothing narrower -- which is the common case and the safe default.

Why it is needed
    The deterministic validator asks "does this page carry substance about the
    topic?" It cannot ask "is this page *about* the narrow thing, or does it
    merely mention it on the way past?" A category hub answers yes to the first
    question and no to the second: the Cards page lists every card family, so a
    question about one family resolves there and stops one hop short of the
    page that actually answers it.

The signal, and why it is not a keyword list
    A question narrows the page it is standing on when it contains a content
    word the page's own URL path does not. "What types of cards do you have?"
    on ``.../Pages/Cards`` narrows nothing -- every content word is already in
    the path. "What types of debit cards do you have?" on the same page has one
    word left over, and that word is the whole point of the question.

    So the test is a set difference between two things the site itself
    provides, and it contains no vocabulary of its own. There is no list of
    product names here, no notion of what a "category" is, and nothing that
    would stop working if this were a site about bicycles.

What it deliberately does not do
    It never decides that a page is *unanswerable*. The navigator uses this to
    *defer* a resolve -- to go and look at the narrower page while keeping the
    verdict it already has -- so a wrong "yes" costs one fetch and a wrong "no"
    costs nothing at all. That asymmetry is why the rules below abstain
    whenever the answer is ambiguous.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import unquote, urlsplit

# Their stopword set, not a second copy of it. It has been tuned against
# scripts/validator_bench.py and it already strips exactly what must not count
# as a narrowing term -- "types", "details", "overview", "rates", "list". A
# private import is the lesser evil next to a divergent duplicate.
from person_b.validation.validator import _distinctive_query_tokens

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _stem(word: str) -> str:
    """Crudest possible plural strip, so "loans" and "loan" are one term.

    Deliberately not a real stemmer: over-stemming would merge terms that the
    site distinguishes, and the only mismatch this needs to survive is the one
    between how people type a category and how the site titles it.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stems(words: set[str]) -> set[str]:
    return {_stem(w) for w in words}


def _path(url: str) -> str:
    return unquote(urlsplit(url or "").path).rstrip("/").lower()


def _path_words(url: str) -> set[str]:
    return _words(_path(url).replace("/", " "))


def narrowing_terms(sub_goal: str, current_url: str) -> list[str]:
    """Content words of the sub-goal that the current page's own path lacks.

    Empty means the question is no narrower than the page -- the ordinary case
    for "what cards do you have?" standing on the cards page, and the reason
    broad questions never trigger a deeper hop.
    """
    here = _stems(_path_words(current_url))
    terms = [_stem(t) for t in _distinctive_query_tokens(sub_goal)]
    return [t for t in dict.fromkeys(terms) if t not in here]


def _describes(link: dict, terms: set[str]) -> bool:
    """Does this link's own description cover every narrowing term?

    The haystack is the label *and* the URL path, because either one alone
    fails on this site: Sitecore uses "More Details" as the visible text for
    tiles whose URL says what they are, and it buries real titles under a
    trailing ``/Details`` segment where only the label says what they are.
    """
    haystack = _stems(_words(link.get("label") or "") | _path_words(link.get("url") or ""))
    return bool(terms) and terms <= haystack


def find_narrowing_link(
    sub_goal: str,
    current_url: str,
    links: list[dict],
    visited: set[str] | frozenset[str] = frozenset(),
    *,
    key_of=None,
) -> dict | None:
    """The one link on this page that is more specifically about the sub-goal.

    ``None`` whenever the answer is not obvious, which is most of the time:

    * the question is no narrower than the page (no terms left over);
    * nothing here covers all the leftover terms;
    * several unrelated pages cover them and none is a child of this page --
      an ambiguous match is a guess, and a guess is what this module exists
      to avoid.

    PDFs are excluded. A PDF is a document, not a narrower section of the site,
    and the ordinary selector already reaches the ones that matter.
    """
    terms = set(narrowing_terms(sub_goal, current_url))
    if not terms:
        return None

    here = _path(current_url)
    children: list[dict] = []
    others: list[dict] = []
    for link in links:
        if link.get("is_pdf"):
            continue
        url = link.get("url") or ""
        if not url or _path(url) == here:
            continue
        key = key_of(url) if key_of else link.get("key")
        if key and key in visited:
            continue
        if not _describes(link, terms):
            continue
        (children if _path(url).startswith(here + "/") else others).append(link)

    if children:
        # The shallowest child is the section itself; anything deeper is a leaf
        # inside it, which the next hop can still reach from there.
        return min(children, key=lambda l: (_path(l["url"]).count("/"), _path(l["url"])))
    if len(others) == 1:
        return others[0]
    return None
