"""Synthesis of validated research evidence into structured answers with explicit claims."""

from typing import Any, Dict, List, Optional, Set
import uuid

from person_b.models import (
    Claim,
    Plan,
    SubGoalStatus,
    SynthesisResult,
)


# --- PATCH 16 (vendor) ------------------------------------------------------
# Table records were rendered cell-by-cell:
#
#     for k, v in rec.items():
#         if k and v and k != v and k.lower() not in ("details", "col_0", "col_1"):
#             statement = f"{t_name}: {k} is {v}."
#
# A record is a *row*: {label_column: row_label, value_column: value}. Iterating
# it cell-wise and then excluding the column literally named "details" -- which
# on this site is the value column -- kept the label and threw the answer away.
# Measured on the Classic card page, the record
#
#     {"Fees and charges": "Issuance", "Details": "EGP 250"}
#
# produced "Fees and charges: Fees and charges is Issuance." -- a row label
# asserted as though it were a value, with no amount anywhere in it. All 41
# table claims on that page had that shape, so a run could report "42 claims,
# 100% verified" while containing not one fee.
#
# Rows are now read as label/value using the table's own header order: the
# first column labels the row, the rest carry values.
def _row_facts(table: Dict[str, Any]) -> List[Dict[str, str]]:
    """Turn one table into ``{label, field, value}`` facts, one per value cell."""
    headers = [h for h in (table.get("headers") or []) if h]
    facts: List[Dict[str, str]] = []
    for rec in table.get("records") or []:
        if not isinstance(rec, dict) or not rec:
            continue
        keys = headers if headers and all(h in rec for h in headers) else list(rec)
        if not keys:
            continue
        if len(keys) == 1:
            # A one-column record is already a {field: value} pair rather than
            # a labelled row, so the key is the label and there is nothing to
            # split off. (Real pages here yield two columns; this shape shows
            # up in hand-built records and must not silently produce nothing.)
            only = keys[0]
            value = str(rec.get(only, "")).strip()
            if value and value != only:
                facts.append({"label": str(only), "field": str(only), "value": value})
            continue
        label_key, value_keys = keys[0], keys[1:]
        label = str(rec.get(label_key, "")).strip()
        if not label:
            continue
        for value_key in value_keys:
            value = str(rec.get(value_key, "")).strip()
            # A cell equal to its own label carries nothing; that identity was
            # the only thing the old ``k != v`` guard was really catching.
            if not value or value == label:
                continue
            facts.append({"label": label, "field": str(value_key), "value": value})
    return facts


def _fact_statement(table_name: str, fact: Dict[str, str], multi_value: bool) -> str:
    """One fact as a sentence that actually contains the value."""
    head = f"{table_name}: {fact['label']}" if table_name else fact["label"]
    if multi_value:
        return f"{head} — {fact['field']}: {fact['value']}."
    return f"{head} — {fact['value']}."
# --- END PATCH 16 ---


def synthesize(
    user_task: str,
    plan: Optional[Plan] = None,
    validated_results: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesize multi-step research findings into an answer structure with explicit claims.

    Aggregates evidence across resolved sub-goals, formats response, and creates Claim objects
    for each factual statement for downstream verification.
    """
    claims: List[Claim] = []
    sources: List[str] = []
    missing_info: List[str] = []
    answer_parts: List[str] = []
    seen_statements: Set[str] = set()

    # Gather sub-goals and evidence
    if plan:
        for sg in plan.sub_goals:
            if sg.status == SubGoalStatus.NOT_AVAILABLE:
                missing_info.append(f"{sg.question}: Information not available on Banque Misr website.")

    if validated_results:
        for res in validated_results:
            extracted = res.get("extracted", {})
            src_url = res.get("source_url")
            if src_url and src_url not in sources:
                sources.append(src_url)

            # 1. Process entities / card list
            entities = extracted.get("entities", [])
            if entities and len(entities) >= 3:
                card_names = [e["name"] for e in entities if e.get("name")]
                if card_names:
                    # --- PATCH 15 (vendor) ----------------------------------
                    # Was: f"Banque Misr offers the following credit cards: ..."
                    # This branch fires on *any* page yielding 3+ entities, so
                    # a loans or accounts run produced a final answer that read
                    # "Banque Misr offers the following credit cards: Personal
                    # Loans, ...". A wrong noun in the delivered answer is worse
                    # than a bland one, and this was the only place the answer
                    # text named a product category it had not established.
                    # --- END PATCH 15 ---
                    statement = f"Banque Misr lists the following: {', '.join(card_names[:8])}."
                    if statement not in seen_statements:
                        seen_statements.add(statement)
                        c = Claim(
                            id=f"c_{uuid.uuid4().hex[:8]}",
                            statement=statement,
                            field="entity_list",  # PATCH 15: was "credit_cards_list"
                            value=", ".join(card_names[:8]),
                            source_url=src_url,
                        )
                        claims.append(c)
                        answer_parts.append(statement)

            # 2. Process tables (fees, limits, installments)   -- PATCH 16
            # 3. Process PDF tables                              -- PATCH 16
            for table in list(extracted.get("tables", [])) + list(extracted.get("pdf_tables", [])):
                t_name = table.get("table_name") or ""
                facts = _row_facts(table)
                multi_value = len({f["field"] for f in facts}) > 1
                for fact in facts:
                    statement = _fact_statement(t_name, fact, multi_value)
                    if statement in seen_statements:
                        continue
                    seen_statements.add(statement)
                    claims.append(
                        Claim(
                            id=f"c_{uuid.uuid4().hex[:8]}",
                            statement=statement,
                            entity=fact["label"],
                            field=fact["field"],
                            value=fact["value"],
                            source_url=src_url,
                        )
                    )
                    # The values are the answer, so they belong in the prose.
                    # They were previously added to ``claims`` and never to
                    # ``answer_parts``, so on any page that also yielded three
                    # entities the entity list became the entire answer and
                    # every extracted figure was dropped.
                    answer_parts.append(statement)

    if not answer_parts:
        if claims:
            top_facts = [c.statement for c in claims[:4]]
            answer_parts.append("Based on Banque Misr official documentation:\n- " + "\n- ".join(top_facts))
        else:
            answer_parts.append(f"Research completed for: {user_task}. No verified facts were retrieved.")

    draft_answer = "\n\n".join(answer_parts)

    result = SynthesisResult(
        task=user_task,
        draft_answer=draft_answer,
        claims=claims,
        sources=sources,
        missing_info=missing_info,
    )

    return result.to_dict()
