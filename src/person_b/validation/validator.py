"""Sub-goal validation against fetched page content and extracted evidence."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from person_b.extraction.extractor import extract_content, preprocess_cleaned_text
from person_b.extraction.normalization import normalize_entity_name, normalize_whitespace
from person_b.models import (
    PageContext,
    SubGoal,
    SubGoalStatus,
    ValidationResult,
    ValidationStatus,
)

_COMMON_QUERY_STOPWORDS = {
    "find", "information", "for", "banque", "misr", "the", "and", "in", "of",
    "to", "a", "an", "what", "is", "details", "overview", "rates", "terms",
    "how", "much", "does", "cost", "available", "get", "list", "all",
}


def _check_entity_relevance(
    target_entity: str,
    raw_text: str,
    extracted_data: Dict[str, Any],
) -> bool:
    """
    Check whether the page body genuinely pertains to the target entity.

    Guards against false positives from misleading first-line titles or nav boilerplate.
    """
    if not target_entity:
        return True

    norm_target = normalize_entity_name(target_entity).lower()

    # 1. Check body lines (excluding nav/footer boilerplate)
    preprocessed = preprocess_cleaned_text(raw_text)
    body_text_lower = preprocessed["focused_text"].lower()

    # Target entity must appear in the focused body text (not just line 1)
    if norm_target not in body_text_lower and target_entity.lower() not in body_text_lower:
        return False

    # Check for negative collision: if page body is clearly Accounts & Deposits,
    # it is not a Classic Credit Card page despite misleading line 1.
    if "classic" in norm_target and "accounts and deposits" in body_text_lower and "fees and charges" not in body_text_lower:
        return False

    return True


def _evaluate_field_coverage(
    target_fields: List[str],
    extracted_data: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Determine which requested target fields are found vs missing in extraction.
    Returns (found_fields, missing_fields).
    """
    if not target_fields or target_fields == ["overview"]:
        return (["overview"], [])

    found = []
    missing = []

    tables = extracted_data.get("tables", [])
    sections = extracted_data.get("sections", {})
    fields = extracted_data.get("fields", {})
    pdf_tables = extracted_data.get("pdf_tables", [])

    all_table_text = " ".join(t.get("table_name", "") + " " + " ".join(t.get("headers", [])) for t in tables).lower()
    all_sections = {k.lower(): v for k, v in sections.items()}

    for tf in target_fields:
        tf_lower = tf.lower()
        if "fee" in tf_lower or "charge" in tf_lower or "cost" in tf_lower:
            if "fee" in all_table_text or "charge" in all_table_text or "fees and charges" in all_sections or any("issuance" in k.lower() for k in fields) or pdf_tables:
                found.append(tf)
            else:
                missing.append(tf)
        elif "benefit" in tf_lower or "feature" in tf_lower:
            if "benefits" in all_sections or "features" in all_sections:
                found.append(tf)
            else:
                missing.append(tf)
        elif "limit" in tf_lower:
            if "limit" in all_table_text or "usage limits" in all_sections or pdf_tables:
                found.append(tf)
            else:
                missing.append(tf)
        elif "interest" in tf_lower or "installment" in tf_lower:
            if "installment" in all_table_text or "tenor" in all_table_text:
                found.append(tf)
            else:
                missing.append(tf)
        else:
            # General field check
            if tf in fields or any(tf_lower in k.lower() for k in fields):
                found.append(tf)
            else:
                missing.append(tf)

    return (found, missing)


def validate(
    sub_goal: Union[str, SubGoal, Dict[str, Any]],
    page_content: Union[str, PageContext],
    *,
    source_url: Optional[str] = None,
    content_type: Optional[str] = None,
    status_code: Optional[int] = None,
    pdf_bytes: Optional[bytes] = None,
    pdf_path: Optional[str] = None,
    pdf_tables: Optional[List[List[List[str]]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate whether page content satisfies the given sub-goal with usable evidence.

    Preserves the mandatory Person A contract:
        validate(sub_goal, page_content) -> {
            "resolved": bool,
            "extracted": dict,
            "reason": str,
        }
    """
    # 1. Resolve sub-goal question, target fields, and entity
    if isinstance(sub_goal, SubGoal):
        sg_question = sub_goal.question
        target_fields = sub_goal.target_fields
        target_entity = sub_goal.metadata.get("target_entity")
        sg_id = sub_goal.id
    elif isinstance(sub_goal, dict):
        sg_question = sub_goal.get("question", str(sub_goal))
        target_fields = sub_goal.get("target_fields", [])
        target_entity = sub_goal.get("metadata", {}).get("target_entity") or sub_goal.get("target_entity")
        sg_id = sub_goal.get("id", "sg_unknown")
    else:
        sg_question = str(sub_goal)
        target_fields = []
        target_entity = None
        sg_id = "sg_unknown"

    # Infer target entity from question if not in metadata (e.g. 'for Classic Credit Card')
    if not target_entity and " for " in sg_question:
        target_entity = sg_question.split(" for ")[-1].strip()

    # 2. Extract content from provided material
    extraction_res = extract_content(
        page_content=page_content,
        target_fields=target_fields,
        source_url=source_url,
        content_type=content_type,
        pdf_bytes=pdf_bytes,
        pdf_path=pdf_path,
        pdf_tables=pdf_tables,
        metadata=metadata,
    )

    url = extraction_res.source_url or source_url
    meta = {**(extraction_res.metadata or {}), **(metadata or {})}
    extracted_data = extraction_res.extracted or {}
    raw_text = page_content.content if isinstance(page_content, PageContext) else str(page_content or "")

    # 3. Handle unreadable PDF
    if extraction_res.status == "unreadable":
        result = ValidationResult(
            resolved=False,
            status=ValidationStatus.UNREADABLE,
            extracted={},
            evidence=[],
            missing_fields=target_fields,
            reason="PDF returned no extractable text/tables (scanned or image-only document).",
            source_url=url,
            warnings=extraction_res.warnings,
            metadata=meta,
        )
        return result.to_dict()

    q_lower = sg_question.lower()

    # 4. Discovery Sub-Goals (e.g. "Find the list of credit cards")
    if "list of" in q_lower or "find all" in q_lower or "credit_cards_list" in target_fields:
        entities = extracted_data.get("entities", [])
        card_entities = [e for e in entities if "card" in e.get("name", "").lower() or "visa" in e.get("name", "").lower() or "master" in e.get("name", "").lower()]
        if len(card_entities) >= 3 or len(entities) >= 5:
            result = ValidationResult(
                resolved=True,
                status=ValidationStatus.RESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=[],
                reason=f"Discovered {len(entities)} product entities on page list.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()
        else:
            result = ValidationResult(
                resolved=False,
                status=ValidationStatus.UNRESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=["product_list"],
                reason="Page does not contain a sufficient product list for discovery.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()

    # 5. Accounts & Deposits category
    if "account" in q_lower or "deposit" in q_lower or "certificate" in q_lower:
        preprocessed = preprocess_cleaned_text(raw_text)
        body_lower = preprocessed["focused_text"].lower()
        if "accounts and deposits" in body_lower or any("account" in e.get("name", "").lower() for e in extracted_data.get("entities", [])):
            result = ValidationResult(
                resolved=True,
                status=ValidationStatus.RESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=[],
                reason="Found Accounts and Deposits offerings and details.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()

    # 6. Consumer Loans category
    if "loan" in q_lower:
        preprocessed = preprocess_cleaned_text(raw_text)
        body_lower = preprocessed["focused_text"].lower()
        if "consumer loans" in body_lower or any("loan" in e.get("name", "").lower() for e in extracted_data.get("entities", [])):
            result = ValidationResult(
                resolved=True,
                status=ValidationStatus.RESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=[],
                reason="Found Consumer Loans offerings and details.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()

    # 7. Fee schedule / Fee hub
    if "fee schedule" in q_lower or "fees hub" in q_lower:
        attachments = [e for e in extracted_data.get("entities", []) if e.get("type") == "attachment"]
        if attachments or extracted_data.get("pdf_tables"):
            result = ValidationResult(
                resolved=True,
                status=ValidationStatus.RESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=[],
                reason="Found fee documents and attachments.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()

    # 8. Entity-Specific Sub-Goals (e.g. Classic Credit Card details)
    if target_entity:
        is_relevant = _check_entity_relevance(target_entity, raw_text, extracted_data)
        if not is_relevant:
            result = ValidationResult(
                resolved=False,
                status=ValidationStatus.UNRESOLVED,
                extracted={},
                evidence=[],
                missing_fields=target_fields,
                reason=f"Page body does not match target entity '{target_entity}'.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()

        # Check field coverage
        found_fields, missing_fields = _evaluate_field_coverage(target_fields, extracted_data)

        if not missing_fields and (extracted_data.get("tables") or extracted_data.get("sections") or extracted_data.get("pdf_tables")):
            result = ValidationResult(
                resolved=True,
                status=ValidationStatus.RESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=[],
                reason=f"Found required evidence for '{target_entity}': {found_fields}.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()
        elif found_fields:
            # Partial evidence
            result = ValidationResult(
                resolved=False,
                status=ValidationStatus.PARTIAL,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=missing_fields,
                reason=f"Partial evidence found for '{target_entity}' ({found_fields}); missing: {missing_fields}.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()
        else:
            result = ValidationResult(
                resolved=False,
                status=ValidationStatus.UNRESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=missing_fields,
                reason=f"Entity '{target_entity}' found on page but none of the requested fields {target_fields} were present.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            )
            return result.to_dict()

    # 9. Generic sub-goal fallback: Check query topic presence in body
    query_tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", sg_question.lower()) if w not in _COMMON_QUERY_STOPWORDS]
    if query_tokens:
        body_text_lower = preprocess_cleaned_text(raw_text)["focused_text"].lower()
        matching_tokens = [w for w in query_tokens if w in body_text_lower]
        # Require substantial topic match in body
        if len(matching_tokens) < max(1, (len(query_tokens) + 1) // 2):
            return ValidationResult(
                resolved=False,
                status=ValidationStatus.UNRESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=target_fields,
                reason=f"Page body does not contain query topics {query_tokens}.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            ).to_dict()

    found_fields, missing_fields = _evaluate_field_coverage(target_fields, extracted_data)
    if extraction_res.evidence and not missing_fields:
        result = ValidationResult(
            resolved=True,
            status=ValidationStatus.RESOLVED,
            extracted=extracted_data,
            evidence=extraction_res.evidence,
            missing_fields=[],
            reason="Found structured evidence satisfying sub-goal.",
            source_url=url,
            warnings=extraction_res.warnings,
            metadata=meta,
        )
    elif extraction_res.evidence:
        result = ValidationResult(
            resolved=False,
            status=ValidationStatus.PARTIAL,
            extracted=extracted_data,
            evidence=extraction_res.evidence,
            missing_fields=missing_fields,
            reason=f"Partial evidence found; missing: {missing_fields}.",
            source_url=url,
            warnings=extraction_res.warnings,
            metadata=meta,
        )
    else:
        result = ValidationResult(
            resolved=False,
            status=ValidationStatus.UNRESOLVED,
            extracted={},
            evidence=[],
            missing_fields=target_fields,
            reason="No relevant structured evidence found on page.",
            source_url=url,
            warnings=extraction_res.warnings,
            metadata=meta,
        )

    return result.to_dict()
