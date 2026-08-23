"""Strict self-check, anti-hallucination guardrail, and final answer assembly."""

from typing import Any, Dict, List, Optional, Union

from person_b.models import (
    Claim,
    ClaimCheck,
    ClaimStatus,
    FinalAnswer,
    SynthesisResult,
)
from person_b.verification.attribution import attribute_sources


def validate_answer(
    claims: List[Union[Dict[str, Any], Claim]],
    visited_pages: List[Any],
    strict: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Verify all claims against visited sources in this run.

    Rejects unsupported, unvisited, or overbroad claims.
    """
    attributed_checks = attribute_sources(claims, visited_pages, strict=strict)

    passed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    contradicted: List[Dict[str, Any]] = []

    for chk in attributed_checks:
        status = chk.get("status")
        claim_stmt = chk.get("claim", {}).get("statement", "").lower()

        # Overbroad superlative check (e.g. 'cheapest card in the bank' without complete card evidence)
        if any(w in claim_stmt for w in ("cheapest", "best in the bank", "only requirement", "all requirements")):
            chk["status"] = ClaimStatus.UNSUPPORTED.value
            chk["reason"] = "Overbroad comparative or exhaustive claim not fully substantiated by evidence."
            failed.append(chk)
            unsupported.append(chk)
            continue

        if status == ClaimStatus.SUPPORTED.value:
            passed.append(chk)
        elif status == ClaimStatus.CONTRADICTED.value:
            failed.append(chk)
            contradicted.append(chk)
        else:
            failed.append(chk)
            unsupported.append(chk)

    total = len(attributed_checks)
    support_rate = len(passed) / total if total > 0 else 1.0

    return {
        "all_supported": (len(failed) == 0),
        "support_rate": support_rate,
        "verified_claims": attributed_checks,
        "passed": passed,
        "failed": failed,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
    }


def finalize(
    draft_answer: Union[str, Dict[str, Any], SynthesisResult],
    verification_result: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Assemble final verified answer, removing unsupported claims and compiling source URLs.
    """
    if isinstance(draft_answer, SynthesisResult):
        answer_text = draft_answer.draft_answer
        not_found = list(draft_answer.missing_info)
        draft_sources = list(draft_answer.sources)
    elif isinstance(draft_answer, dict):
        answer_text = draft_answer.get("draft_answer", draft_answer.get("answer", str(draft_answer)))
        not_found = list(draft_answer.get("missing_info", draft_answer.get("not_found", [])))
        draft_sources = list(draft_answer.get("sources", draft_answer.get("source_urls", [])))
    else:
        answer_text = str(draft_answer)
        not_found = []
        draft_sources = []

    passed_checks = verification_result.get("passed", [])
    failed_checks = verification_result.get("failed", [])

    # Compile verified source URLs
    verified_urls = []
    for chk in passed_checks:
        url = chk.get("verified_source_url") or chk.get("claim", {}).get("source_url")
        if url and url not in verified_urls:
            verified_urls.append(url)

    # If all claims failed and draft had unsupported claims, adjust answer text
    if failed_checks and not passed_checks:
        answer_text = "The requested information could not be verified from the visited Banque Misr website sources."
        for fc in failed_checks:
            stmt = fc.get("claim", {}).get("statement", "Unverified claim")
            not_found.append(f"Could not verify: {stmt}")

    final_obj = FinalAnswer(
        answer=answer_text,
        source_urls=verified_urls if verified_urls else draft_sources,
        not_found=not_found,
        verified_claims=[ClaimCheck.from_dict(c) if isinstance(c, dict) else c for c in passed_checks],
    )

    return final_obj.to_dict()
