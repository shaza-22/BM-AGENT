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
    # --- PATCH 3 (vendor) ---------------------------------------------------
    # Added: generic question scaffolding and this planner's own padding.
    #
    # Measured on scripts/validator_bench.py, the words a *correct* page failed
    # to match were almost entirely scaffolding -- "offer", "tell", "about",
    # "show", "document", "are" -- while the words a *wrong* page failed to
    # match were the specific nouns that carry the question ("cryptocurrency",
    # "martian", "dog", "student", "islamic"). Treating scaffolding as content
    # is what made the topic test fire on the wrong axis: it penalised phrasing
    # instead of subject matter.
    #
    # "offerings"/"offering" are here because plan_task() injects them into the
    # rewritten question, so they describe the planner, not the user.
    "offer", "offers", "offered", "offering", "offerings",
    "tell", "about", "show", "give", "need", "want", "know", "please",
    "are", "was", "were", "can", "could", "would", "should", "will",
    "with", "from", "any", "some", "there", "their", "this", "that",
    "have", "has", "had", "you", "your", "our", "its", "they", "them",
    "which", "who", "whom", "when", "why", "where", "banking", "bank",
    # Generic classifier words. "Find Banque Misr accounts and deposit types"
    # is asking about accounts and deposits; "types" adds no subject matter,
    # and requiring it on the page rejected the very hub that answers the
    # question (their test_validation_misleading_title_defense covers exactly
    # this). Same argument as the scaffolding above.
    "type", "types", "kind", "kinds", "sort", "sorts", "option", "options",
    # --- END PATCH 3 ---
}


# --- PATCH 4 (vendor) -------------------------------------------------------
# Extracted from the generic fallback (was inline at step 9) so the same topic
# test can gate the category branches at steps 4-7, which previously granted
# resolved=True on category membership alone. See _question_topics_present.
def _distinctive_query_tokens(question: str) -> List[str]:
    """Content words of a question, in order, without repeats or scaffolding."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", (question or "").lower())
    return list(dict.fromkeys(w for w in words if w not in _COMMON_QUERY_STOPWORDS))


# How many content words of a question may be absent from the page and the page
# still count as on-topic. 0 = every content word must appear.
#
# Chosen by measurement, not by feel: scripts/validator_bench.py sweeps this
# against 23 labelled (task, page) pairs. The sweep is reproduced in
# src/person_b/PATCHES.md. 0 is the only value with no false positives, and
# every larger value buys back the same single false negative at the cost of
# several false positives -- a false positive stops navigation on a page that
# cannot answer the question, which is strictly worse than one extra hop.
_TOPIC_MISS_TOLERANCE = 0


def _question_topics_present(question: str, raw_text: str) -> Tuple[bool, List[str]]:
    """Is every content word of the question actually on this page?

    Returns ``(ok, missing_tokens)``. A question with no content words at all
    (pure scaffolding) is treated as satisfied -- there is nothing to check,
    and the caller's own evidence tests still apply.

    The test runs against ``preprocess_cleaned_text``'s focused body, so on an
    interior page a word appearing only in the global navigation does not count
    as being "on the page".

    Note the limit of that: the stripper locates the body by finding a
    breadcrumb, and the homepage has none, so on the homepage focused_text is
    the whole page and nav words *do* match here. The homepage is kept from
    answering everything by the substance requirement in
    ``_evaluate_field_coverage`` (PATCH 5), not by this gate.
    """
    tokens = _distinctive_query_tokens(question)
    if not tokens:
        return (True, [])
    body_lower = preprocess_cleaned_text(raw_text)["focused_text"].lower()
    missing = [w for w in tokens if w not in body_lower]
    return (len(missing) <= _TOPIC_MISS_TOLERANCE, missing)
# --- END PATCH 4 ---


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
    # --- PATCH 5 (vendor) ---------------------------------------------------
    # Was:
    #     if not target_fields or target_fields == ["overview"]:
    #         return (["overview"], [])
    #
    # This was the only branch in the function that never looked at the page.
    # "overview" is the planner's default whenever the task names no specific
    # dimension, so for most questions coverage was declared complete before
    # any evidence was inspected. The caller then resolves on
    # ``evidence and not missing_fields`` -- and *any* page yields some
    # evidence, so any page answered any overview question. That is the
    # mechanism behind the homepage resolving "how do I open an account?":
    # its nav tiles produced 6 evidence items and coverage waved them through.
    #
    # An overview is now satisfied by substance a reader would recognise as
    # page content -- named entities, sections, or tables. Measured across the
    # fixtures, content pages yield 6-12 entities each while the homepage
    # yields 0, so this separates article pages from pure navigation without
    # naming a single product category.
    if not target_fields or target_fields == ["overview"]:
        # "tables" is deliberately not accepted on its own. The site's currency
        # widget is parsed as a table on every page including the homepage --
        # its headers are the unrendered placeholders ["{{fromCurrency}}",
        # "Cash", "Transfer"] -- so a bare table with no named entity and no
        # section anywhere on the page is chrome, not an overview. Pages that
        # do carry real tables (the Classic card page has three) also carry
        # entities, so nothing substantive is lost.
        has_substance = bool(
            extracted_data.get("entities")
            or extracted_data.get("sections")
            or extracted_data.get("pdf_tables")
        )
        return ((["overview"], []) if has_substance else ([], ["overview"]))
    # --- END PATCH 5 ---

    found = []
    missing = []

    tables = extracted_data.get("tables", [])
    sections = extracted_data.get("sections", {})
    fields = extracted_data.get("fields", {})
    pdf_tables = extracted_data.get("pdf_tables", [])

    all_table_text = " ".join(t.get("table_name", "") + " " + " ".join(t.get("headers", [])) for t in tables).lower()
    all_sections = {k.lower(): v for k, v in sections.items()}

    # --- PATCH 9 (vendor) ---------------------------------------------------
    # Coverage consulted table headers, sections, fields and pdf_tables -- but
    # never the names of the entities the extractor had just pulled off the
    # page. A hub whose answer *is* a set of documents therefore scored zero:
    # the fees hub carries entities like "Banque Misr payment cards Fees ,
    # Limits and commission" and "BM Online banking transfer limits for
    # individuals", and was still reported as missing "fees" and "limits".
    #
    # Entity names now join the searchable text for every branch below, so the
    # rule is uniform rather than a new special case. This only ever adds a
    # place to find a field; it cannot mark a found field missing.
    all_entity_text = " ".join(
        str(e.get("name", "")) for e in extracted_data.get("entities", []) if isinstance(e, dict)
    ).lower()
    all_table_text = f"{all_table_text} {all_entity_text}".strip()
    # --- END PATCH 9 ---

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
            # --- PATCH 11 (vendor) ------------------------------------------
            # Added "interest"/"rate" to the accepted markers. The field is
            # named "interest_rate" but the test looked only for "installment"
            # and "tenor", so a page headed "Deposits Interest Rates for
            # individual customers" was reported as missing its interest rate.
            # --- END PATCH 11 ---
            if ("installment" in all_table_text or "tenor" in all_table_text
                    or "interest" in all_table_text or "rate" in all_table_text):
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
    # --- PATCH 8 (vendor) ---------------------------------------------------
    # Was: target_entity = sg_question.split(" for ")[-1].strip()
    #
    # The tail of a rewritten question keeps its leading article and trailing
    # punctuation, so "What is the annual fee for the Classic credit card?"
    # yielded the entity "the Classic credit card?" -- which of course appears
    # on no page, so _check_entity_relevance rejected the very page holding the
    # answer with "Page body does not match target entity". Measured: this
    # alone turned a correct fee page into UNRESOLVED.
    #
    # Also switched to partition() so only the *first* " for " splits the
    # string; rsplit on a question containing two "for"s truncated the entity.
    if not target_entity and " for " in sg_question:
        target_entity = sg_question.split(" for ", 1)[-1].strip()
        target_entity = target_entity.strip(" \t?.!,:;\"'")
        for _article in ("the ", "a ", "an ", "my ", "your "):
            if target_entity.lower().startswith(_article):
                target_entity = target_entity[len(_article):]
                break
    # --- END PATCH 8 ---

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
        # --- PATCH 6 (vendor) -----------------------------------------------
        # Added the _question_topics_present gate. The condition below is
        # ``>= 3 matching entities OR >= 5 entities of any kind``, and the
        # second half asks nothing about the question: every fixture content
        # page carries 6-12 entities, so *any* discovery question resolved on
        # *any* content page. Measured: "what credit cards does Banque Misr
        # offer?" resolved on the accounts-and-deposits page, reason
        # "Discovered 8 product entities on page list."
        # The branch's own entity counting is left exactly as it was; it may
        # now only fire on a page that actually discusses what was asked.
        topics_ok, missing_topics = _question_topics_present(sg_question, raw_text)
        if not topics_ok:
            return ValidationResult(
                resolved=False,
                status=ValidationStatus.UNRESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=target_fields,
                reason=f"Page lists products but does not discuss {missing_topics}.",
                source_url=url,
                warnings=extraction_res.warnings,
                metadata=meta,
            ).to_dict()
        # --- END PATCH 6 ---
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

    # --- PATCH 7 (vendor) ---------------------------------------------------
    # The three category branches below (Accounts & Deposits, Consumer Loans,
    # Fee schedule) each granted resolved=True on a category-phrase match with
    # no test that the question's actual subject was on the page. Measured, that
    # made them rubber stamps: "can I open a joint account with my dog?",
    # "what is the interest rate on a Martian savings account?" and "does
    # Banque Misr offer a cryptocurrency deposit account?" all resolved on the
    # accounts hub, reason "Found Accounts and Deposits offerings and details."
    # Any question containing a category word resolved against any page in that
    # category, so navigation stopped at the first hub it reached.
    #
    # One gate, computed once, guards all three. Their category conditions are
    # untouched -- membership is still necessary, it is simply no longer
    # sufficient.
    _topics_ok, _missing_topics = _question_topics_present(sg_question, raw_text)

    def _off_topic_result() -> Dict[str, Any]:
        return ValidationResult(
            resolved=False,
            status=ValidationStatus.UNRESOLVED,
            extracted=extracted_data,
            evidence=extraction_res.evidence,
            missing_fields=target_fields,
            reason=(
                "Page is in the right category but its body does not discuss "
                f"{_missing_topics}."
            ),
            source_url=url,
            warnings=extraction_res.warnings,
            metadata=meta,
        ).to_dict()
    # --- END PATCH 7 ---

    # 5. Accounts & Deposits category
    if "account" in q_lower or "deposit" in q_lower or "certificate" in q_lower:
        preprocessed = preprocess_cleaned_text(raw_text)
        body_lower = preprocessed["focused_text"].lower()
        if "accounts and deposits" in body_lower or any("account" in e.get("name", "").lower() for e in extracted_data.get("entities", [])):
            if not _topics_ok:  # PATCH 7
                return _off_topic_result()
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
            if not _topics_ok:  # PATCH 7
                return _off_topic_result()
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
            if not _topics_ok:  # PATCH 7
                return _off_topic_result()
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
    # --- PATCH 12 (vendor) ---------------------------------------------------
    # Was an inline "at least half the query tokens appear in the body" test,
    # a second, different topic rule living alongside the one now used at
    # steps 4-7. Two rules that can disagree in one module is how a page ends
    # up accepted by one branch and rejected by another for the same question.
    # This now calls the same helper, so the tolerance knob governs the whole
    # module and the sweep in PATCHES.md measures one thing rather than two.
    # --- END PATCH 12 ---
    if _distinctive_query_tokens(sg_question):
        topics_ok, missing_topics = _question_topics_present(sg_question, raw_text)
        if not topics_ok:
            return ValidationResult(
                resolved=False,
                status=ValidationStatus.UNRESOLVED,
                extracted=extracted_data,
                evidence=extraction_res.evidence,
                missing_fields=target_fields,
                reason=f"Page body does not contain query topics {missing_topics}.",
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
