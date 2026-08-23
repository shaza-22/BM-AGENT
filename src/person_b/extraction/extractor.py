"""Generic content extraction pipeline for cleaned text, pipe tables, and PDFs."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from person_b.extraction.normalization import (
    extract_currency_value,
    normalize_entity_name,
    normalize_field_value,
    normalize_whitespace,
)
from person_b.extraction.pdf_tables import extract_pdf_tables
from person_b.extraction.text_tables import parse_text_tables
from person_b.models import (
    Evidence,
    EvidenceLocation,
    ExtractionLocationType,
    ExtractionResult,
    PageContext,
)


# --- PATCH 1 (vendor) -------------------------------------------------------
# Removed _NAV_START_MARKERS. Two reasons:
#
#   1. It was dead code. The list was defined here but never read -- only
#      _FOOTER_START_MARKERS is used by preprocess_cleaned_text(), so the
#      nav half of the intended boilerplate stripping was never wired up.
#   2. Its first entry was the literal string "ie ??? ????" -- a mangled
#      Arabic phrase, i.e. text that had already lost its encoding before it
#      was pasted in as a constant. Matching on it could only ever be
#      accidental, and it is the same mojibake that makes this module's
#      Arabic handling a known gap.
#
# Nothing referenced it, so deleting it is behaviour-preserving; it is
# recorded rather than silently dropped so a future drop from Person B can be
# checked for the same constant.
# --- END PATCH 1 ---

_FOOTER_START_MARKERS = [
    "link has been copied",
    "quick links",
    "bm cards offers",
    "all rights reserved for banquemisr",
]


def preprocess_cleaned_text(text: str) -> Dict[str, Any]:
    """
    Preprocess cleaned page text to isolate the main body content from navigation/footer boilerplate.

    Preserves raw text for complete snippet evidence while providing focused body text.
    """
    if not text:
        return {"raw_text": "", "body_lines": [], "focused_text": "", "headings": []}

    lines = [l.rstrip() for l in text.splitlines()]
    body_lines: List[Tuple[int, str]] = []
    in_footer = False

    # Identify breadcrumb / content start
    breadcrumb_indices = []
    for idx, line in enumerate(lines[:60]):
        stripped = line.strip().lower()
        if stripped in ("smes", "retail banking", "pages", "cards", "accounts and deposits", "consumer loans", "fees and rates", "fees"):
            breadcrumb_indices.append(idx)

    body_start_idx = breadcrumb_indices[0] if breadcrumb_indices else 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()

        # Check for footer start
        if any(marker in lower for marker in _FOOTER_START_MARKERS):
            in_footer = True

        if idx >= body_start_idx and not in_footer:
            body_lines.append((idx, stripped))

    focused_text = "\n".join(l for _, l in body_lines)
    return {
        "raw_text": text,
        "body_lines": body_lines,
        "focused_text": focused_text,
    }


def extract_entities_and_lists(
    lines: List[Tuple[int, str]],
    source_url: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Evidence]]:
    """
    Extract product/entity cards, offerings, and lists from body text lines.

    Recognizes patterns like:
      <Entity Name>
      [Optional Description]
      More Details / View more details
    or:
      <Attachment Name>
      Download
    """
    entities: List[Dict[str, Any]] = []
    evidence_list: List[Evidence] = []
    seen_names: Set[str] = set()

    for i in range(len(lines)):
        line_no, text = lines[i]
        stripped = text.strip()

        # Check for 'More Details' / 'View more details' trigger
        if stripped.lower() in ("more details", "view more details") and i > 0:
            candidate_line_no, candidate_text = lines[i - 1]
            candidate_text = normalize_whitespace(candidate_text)

            entity_name = candidate_text
            description = None

            # If candidate_text is a long paragraph and the preceding line is a short title
            if len(candidate_text) > 45 and i > 1:
                prev_line_no, prev_text = lines[i - 2]
                prev_text = normalize_whitespace(prev_text)
                if 2 < len(prev_text) <= 45 and "|" not in prev_text and prev_text.lower() not in ("more details", "view more details", "download", "link has been copied"):
                    entity_name = prev_text
                    description = candidate_text
                    candidate_line_no = prev_line_no

            # Validate not a table row or generic keyword
            if entity_name and "|" not in entity_name and entity_name.lower() not in ("more", "download", "link has been copied"):
                norm_name = normalize_entity_name(entity_name)
                if norm_name and norm_name not in seen_names:
                    seen_names.add(norm_name)

                    loc = EvidenceLocation(
                        type=ExtractionLocationType.LIST,
                        row_index=candidate_line_no,
                        details={"trigger": stripped, "raw_name": entity_name},
                    )
                    ev = Evidence(
                        field="product_name",
                        value=norm_name,
                        source_url=source_url,
                        evidence_text=f"{entity_name}\n{description or stripped}",
                        location=loc,
                        confidence=0.95,
                    )
                    evidence_list.append(ev)

                    entities.append({
                        "name": norm_name,
                        "raw_name": entity_name,
                        "description": description,
                        "line_number": candidate_line_no + 1,
                    })

        # Check for 'Download' attachment trigger
        elif stripped.lower() == "download" and i > 0:
            candidate_line_no, candidate_title = lines[i - 1]
            candidate_title = normalize_whitespace(candidate_title)

            if candidate_title and "|" not in candidate_title:
                loc = EvidenceLocation(
                    type=ExtractionLocationType.LIST,
                    row_index=candidate_line_no,
                    details={"type": "attachment", "raw_title": candidate_title},
                )
                ev = Evidence(
                    field="attachment_title",
                    value=candidate_title,
                    source_url=source_url,
                    evidence_text=f"{candidate_title}\nDownload",
                    location=loc,
                    confidence=0.95,
                )
                evidence_list.append(ev)

                entities.append({
                    "name": candidate_title,
                    "type": "attachment",
                    "line_number": candidate_line_no + 1,
                })

    return entities, evidence_list


def extract_benefits_and_sections(
    lines: List[Tuple[int, str]],
    source_url: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], List[Evidence]]:
    """Extract bullet-point benefits and key sections from product page."""
    sections: Dict[str, List[str]] = {}
    evidence_list: List[Evidence] = []

    current_section = None
    current_items: List[str] = []

    for line_no, text in lines:
        stripped = text.strip()
        if not stripped or "|" in stripped:
            continue

        lower = stripped.lower()

        # Known section headers
        if lower in ("benefits", "features", "advantages", "credit limit", "usage limits", "fees and charges"):
            if current_section and current_items:
                sections[current_section] = list(current_items)
            current_section = stripped
            current_items = []
            continue

        if current_section and len(stripped) > 5 and stripped.lower() not in ("more details", "view more details", "download"):
            current_items.append(stripped)
            loc = EvidenceLocation(
                type=ExtractionLocationType.PROSE,
                row_index=line_no,
                details={"section": current_section},
            )
            ev = Evidence(
                field=f"{current_section.lower()}_item",
                value=stripped,
                source_url=source_url,
                evidence_text=stripped,
                location=loc,
                confidence=0.9,
            )
            evidence_list.append(ev)

    if current_section and current_items:
        sections[current_section] = current_items

    return sections, evidence_list


def extract_content(
    page_content: Union[str, PageContext],
    target_fields: Optional[List[str]] = None,
    *,
    source_url: Optional[str] = None,
    content_type: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    pdf_path: Optional[str] = None,
    pdf_tables: Optional[List[List[List[str]]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ExtractionResult:
    """
    Generic structured content extraction across HTML/text and PDF documents.

    Extracts:
      - Pipe-delimited text tables into structured records
      - PDF table matrices preserving multi-column relationships
      - Product and offering lists
      - Benefit items and sections
      - Attachment and download document titles
      - Tying every extracted value to explicit Evidence provenance.
    """
    # 1. Resolve context parameters
    if isinstance(page_content, PageContext):
        raw_text = page_content.content or ""
        url = source_url or page_content.source_url
        c_type = content_type or page_content.content_type
        p_bytes = pdf_bytes or page_content.pdf_bytes
        p_path = pdf_path or page_content.pdf_path
        p_tables = pdf_tables or page_content.pdf_tables
        meta = {**(page_content.metadata or {}), **(metadata or {})}
    else:
        raw_text = str(page_content or "")
        url = source_url
        c_type = content_type or "text"
        p_bytes = pdf_bytes
        p_path = pdf_path
        p_tables = pdf_tables
        meta = metadata or {}

    extracted_data: Dict[str, Any] = {}
    all_evidence: List[Evidence] = []
    warnings: List[str] = []

    # 2. Check for empty text without PDF
    is_pdf_content = (c_type == "pdf") or (p_bytes is not None) or (p_path is not None) or (p_tables is not None)
    if not is_pdf_content and not raw_text.strip():
        return ExtractionResult(
            status="empty",
            extracted={},
            evidence=[],
            missing_fields=target_fields or [],
            warnings=[],
            source_url=url,
            metadata=meta,
        )
    missing_fields: List[str] = []

    # 2. Handle PDF extraction
    if c_type == "pdf" or p_bytes is not None or p_path is not None:
        pdf_input = p_bytes if p_bytes is not None else p_path
        if pdf_input:
            pdf_result = extract_pdf_tables(pdf_input, source_url=url)
            extracted_data["pdf_tables"] = pdf_result["tables"]
            extracted_data["total_pdf_pages"] = pdf_result["total_pages"]
            all_evidence.extend(pdf_result["evidence"])

            if pdf_result.get("is_unreadable"):
                return ExtractionResult(
                    status="unreadable",
                    extracted={},
                    evidence=[],
                    missing_fields=target_fields or [],
                    warnings=pdf_result.get("warnings", ["PDF is unreadable or image-only."]),
                    source_url=url,
                    metadata=meta,
                )

            if pdf_result.get("warnings"):
                warnings.extend(pdf_result["warnings"])

    # 3. Handle Text / HTML extraction
    if raw_text:
        preprocessed = preprocess_cleaned_text(raw_text)
        body_lines = preprocessed["body_lines"]

        # Parse text tables
        parsed_tables = parse_text_tables(raw_text, source_url=url)
        extracted_data["tables"] = parsed_tables
        for t in parsed_tables:
            all_evidence.extend(t.get("evidence", []))

        # Extract entities / product lists
        entities, entity_ev = extract_entities_and_lists(body_lines, source_url=url)
        extracted_data["entities"] = entities
        all_evidence.extend(entity_ev)

        # Extract benefits / sections
        sections, sec_ev = extract_benefits_and_sections(body_lines, source_url=url)
        extracted_data["sections"] = sections
        all_evidence.extend(sec_ev)

        # Build flat field mappings from tables and sections for convenience
        fields_map: Dict[str, Any] = {}
        for t in parsed_tables:
            for rec in t.get("records", []):
                for k, v in rec.items():
                    if k and v:
                        fields_map[k] = v
        extracted_data["fields"] = fields_map

    # 4. Check for duplicate/conflicting evidence
    evidence_by_field: Dict[str, List[Evidence]] = {}
    for ev in all_evidence:
        evidence_by_field.setdefault(ev.field, []).append(ev)

    for field_name, ev_list in evidence_by_field.items():
        if len(ev_list) > 1:
            values = set(normalize_whitespace(str(e.value)) for e in ev_list)
            if len(values) > 1:
                warnings.append(f"Multiple conflicting values found for field '{field_name}': {list(values)}")

    # 5. Check target fields if requested
    if target_fields:
        available_field_names = {ev.field.lower() for ev in all_evidence}
        for tf in target_fields:
            tf_lower = tf.lower()
            if not any(tf_lower in f for f in available_field_names) and tf not in extracted_data.get("fields", {}):
                missing_fields.append(tf)

    status = "success" if (extracted_data.get("tables") or extracted_data.get("entities") or extracted_data.get("pdf_tables") or extracted_data.get("sections")) else "empty"

    return ExtractionResult(
        status=status,
        extracted=extracted_data,
        evidence=all_evidence,
        missing_fields=missing_fields,
        warnings=warnings,
        source_url=url,
        metadata=meta,
    )
