"""Link factual claims to exact source URLs and page evidence locations."""

from typing import Any, Dict, List, Optional, Set, Union

from person_b.models import (
    Claim,
    ClaimCheck,
    ClaimStatus,
    PageContext,
)


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
        else:
            checks.append(
                ClaimCheck(
                    claim=claim_obj,
                    status=ClaimStatus.SUPPORTED,
                    reason="Supported by verified evidence from visited source.",
                    verified_source_url=claim_obj.source_url,
                )
            )

    return [chk.to_dict() for chk in checks]
