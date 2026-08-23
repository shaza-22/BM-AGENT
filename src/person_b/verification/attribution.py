"""Link factual claims to exact source URLs and page evidence locations."""

from typing import Any, Dict, List, Optional, Set, Union

from person_b.models import (
    Claim,
    ClaimCheck,
    ClaimStatus,
    PageContext,
)


# --- PATCH 17 (vendor) ------------------------------------------------------
# attribute_sources decided SUPPORTED purely on "is this claim's source_url in
# the visited set". It never compared the claim's text to the page. Since
# synthesize() stamps every claim it builds with the URL of the page it was
# built from, every claim was supported by construction: a run could report
# "42 claims, 100% verified" meaning only "42 claims cite a page we fetched".
#
# That makes the whole verification story nominal, and it is the number the UI
# puts a bar around -- so it is patched here rather than worked around.
#
# The check added is textual and deterministic: a claim's value must actually
# occur in the text of the page it cites. When the caller passes pages that
# carry their content (PageContext, or a dict with url/text), that check runs.
# When it passes bare URL strings there is no text to check against and the
# behaviour is exactly as before, so callers that cannot supply content -- their
# own test suite among them -- are unaffected. Strictly stronger where it can
# be, never weaker.
import re as _re


def _normalise_for_match(text: str) -> str:
    """Collapse whitespace and case so a value matches its rendering on the page."""
    return _re.sub(r"\s+", " ", (text or "")).strip().casefold()


def _page_texts(visited_pages: List[Any]) -> Dict[str, str]:
    """Map visited URL -> page text, for those pages that carry their content."""
    texts: Dict[str, str] = {}
    for p in visited_pages:
        url = content = None
        if isinstance(p, PageContext):
            url, content = p.source_url, p.content
        elif isinstance(p, dict):
            url = p.get("url") or p.get("source_url")
            content = p.get("text") or p.get("content")
        if url and content:
            texts[url.strip().lower()] = _normalise_for_match(content)
    return texts


def _claim_text_found(claim: Claim, page_text: str) -> bool:
    """Is what this claim asserts actually on the page it cites?

    The value is what carries the assertion -- "EGP 250" is the part that can
    be wrong -- so that is what must be present. A claim with no value falls
    back to its evidence text, and a claim with neither cannot be checked
    textually and is left to the URL rule.
    """
    for candidate in (claim.value, claim.evidence_text):
        if not candidate:
            continue
        needle = _normalise_for_match(str(candidate))
        if not needle:
            continue
        # A list-valued claim ("A, B, C") is supported when each of its parts
        # is on the page; requiring the joined string to appear verbatim would
        # fail on any page that lists them in a different order.
        parts = [q.strip() for q in needle.split(",")] if "," in needle else [needle]
        parts = [q for q in parts if len(q) >= 2]
        return bool(parts) and all(q in page_text for q in parts)
    return True  # nothing quotable to check; the URL rule still applies
# --- END PATCH 17 ---


def _extract_visited_urls(visited_pages: List[Any]) -> Set[str]:
    """Extract normalized set of URLs from visited pages list."""
    urls: Set[str] = set()
    for p in visited_pages:
        if isinstance(p, PageContext) and p.source_url:
            urls.add(p.source_url.strip().lower())
        elif isinstance(p, dict) and p.get("url"):
            urls.add(p["url"].strip().lower())
        elif isinstance(p, dict) and p.get("source_url"):
            urls.add(p["source_url"].strip().lower())
        elif isinstance(p, str):
            urls.add(p.strip().lower())
    return urls


def attribute_sources(
    claims: List[Union[Dict[str, Any], Claim]],
    visited_pages: List[Any],
    strict: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Map each claim to verified evidence from visited pages during this run.

    In strict mode, claims referencing unvisited or missing source URLs fail attribution.
    """
    visited_urls = _extract_visited_urls(visited_pages)
    page_texts = _page_texts(visited_pages)   # PATCH 17
    checks: List[ClaimCheck] = []

    for c in claims:
        claim_obj = Claim.from_dict(c) if isinstance(c, dict) else c
        claim_url = claim_obj.source_url.strip().lower() if claim_obj.source_url else None

        if not claim_url:
            checks.append(
                ClaimCheck(
                    claim=claim_obj,
                    status=ClaimStatus.UNSUPPORTED,
                    reason="Claim lacks a source URL.",
                    verified_source_url=None,
                )
            )
        elif strict and claim_url not in visited_urls:
            checks.append(
                ClaimCheck(
                    claim=claim_obj,
                    status=ClaimStatus.UNSUPPORTED,
                    reason=f"Source URL '{claim_obj.source_url}' was not visited during this run.",
                    verified_source_url=None,
                )
            )
        elif claim_url in page_texts and not _claim_text_found(claim_obj, page_texts[claim_url]):
            # PATCH 17: the page was visited, but it does not say this.
            checks.append(
                ClaimCheck(
                    claim=claim_obj,
                    status=ClaimStatus.UNSUPPORTED,
                    reason=(
                        "Claim cites a visited page, but its value does not appear "
                        f"in that page's text: {str(claim_obj.value)[:120]!r}"
                    ),
                    verified_source_url=None,
                )
            )
        else:
            checks.append(
                ClaimCheck(
                    claim=claim_obj,
                    status=ClaimStatus.SUPPORTED,
                    reason=(
                        "Value found in the text of the cited page."
                        if claim_url in page_texts
                        else "Supported by verified evidence from visited source."
                    ),
                    verified_source_url=claim_obj.source_url,
                )
            )

    return [chk.to_dict() for chk in checks]
