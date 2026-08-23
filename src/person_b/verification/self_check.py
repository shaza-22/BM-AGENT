"""Strict self-check, anti-hallucination guardrail, and final answer assembly."""

from typing import Any, Dict, List, Optional, Tuple, Union

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


# --- PATCH 14 (vendor) ------------------------------------------------------
# finalize() filtered *source URLs* down to the verified ones, but only
# replaced the answer text when **every** claim failed. With 3 of 4 claims
# verified, the unsupported fourth stayed in the prose with a list of verified
# sources printed beneath it -- which reads as attribution for a claim nothing
# supports. That is the exact shape of citation-laundering the verification
# step exists to prevent, so the guarantee has to hold sentence by sentence,
# not just in the all-or-nothing case.
#
# synthesize() builds draft_answer out of claim statements verbatim, one per
# paragraph or one per "- " bullet, so a failed claim can be located in the
# prose by its own statement text and removed.
def _strip_unsupported_prose(answer_text: str, failed_checks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Remove the prose carrying each failed claim. Returns (text, removed_statements)."""
    statements = []
    for chk in failed_checks:
        stmt = (chk.get("claim", {}) or {}).get("statement", "")
        if isinstance(stmt, str) and stmt.strip():
            statements.append(stmt.strip())
    if not statements or not answer_text:
        return (answer_text, [])

    removed: List[str] = []
    kept_blocks: List[str] = []
    for block in answer_text.split("\n\n"):
        kept_lines: List[str] = []
        for line in block.split("\n"):
            hit = next((s for s in statements if s and s in line), None)
            if hit is None:
                kept_lines.append(line)
            elif hit not in removed:
                removed.append(hit)
        remainder = "\n".join(kept_lines).strip()
        # A bullet list whose every item was struck leaves only its lead-in
        # ("Based on Banque Misr official documentation:"), which would stand
        # over nothing. Drop the orphaned lead-in with it.
        if remainder and not remainder.endswith(":"):
            kept_blocks.append(remainder)
        elif remainder.endswith(":") and len(kept_lines) > 1:
            kept_blocks.append(remainder)
    return ("\n\n".join(b for b in kept_blocks if b.strip()), removed)
# --- END PATCH 14 ---


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
    final_metadata = kwargs.get("metadata") or {}

    # Compile verified source URLs
    verified_urls = []
    for chk in passed_checks:
        url = chk.get("verified_source_url") or chk.get("claim", {}).get("source_url")
        if url and url not in verified_urls:
            verified_urls.append(url)

    # --- PATCH 14 (vendor) --------------------------------------------------
    # Strike the prose for every failed claim, not only in the all-failed case.
    # Each struck statement is reported in not_found, so the answer loses the
    # sentence and the user is told what was dropped rather than the claim
    # vanishing silently.
    answer_text, removed_statements = _strip_unsupported_prose(answer_text, failed_checks)
    for stmt in removed_statements:
        not_found.append(f"Could not verify: {stmt}")

    if failed_checks and not passed_checks:
        answer_text = "The requested information could not be verified from the visited Banque Misr website sources."
        for fc in failed_checks:
            stmt = (fc.get("claim", {}) or {}).get("statement", "Unverified claim")
            entry = f"Could not verify: {stmt}"
            if entry not in not_found:
                not_found.append(entry)
    elif not answer_text.strip():
        # Everything that survived attribution was struck as prose.
        answer_text = "The requested information could not be verified from the visited Banque Misr website sources."
    # --- END PATCH 14 ---

    final_obj = FinalAnswer(
        answer=answer_text,
        source_urls=verified_urls if verified_urls else draft_sources,
        not_found=not_found,
        verified_claims=[ClaimCheck.from_dict(c) if isinstance(c, dict) else c for c in passed_checks],
        # --- PATCH 14 (vendor) ----------------------------------------------
        # Carry the verification numbers out with the answer. They were
        # computed by validate_answer() and then dropped on the floor, so a
        # caller had no way to show how much of the answer was actually
        # supported -- which is the visible evidence that verification ran.
        metadata={
            **(final_metadata or {}),
            "support_rate": verification_result.get("support_rate", 1.0),
            "claims_total": len(passed_checks) + len(failed_checks),
            "claims_supported": len(passed_checks),
            "claims_unsupported": len(verification_result.get("unsupported_claims", [])),
            "claims_contradicted": len(verification_result.get("contradicted_claims", [])),
            "prose_removed": removed_statements,
        },
        # --- END PATCH 14 ---
    )

    return final_obj.to_dict()
