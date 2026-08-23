"""Synthesis of validated research evidence into structured answers with explicit claims."""

from typing import Any, Dict, List, Optional, Set
import uuid

from person_b.models import (
    Claim,
    Plan,
    SubGoalStatus,
    SynthesisResult,
)


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
                    statement = f"Banque Misr offers the following credit cards: {', '.join(card_names[:8])}."
                    if statement not in seen_statements:
                        seen_statements.add(statement)
                        c = Claim(
                            id=f"c_{uuid.uuid4().hex[:8]}",
                            statement=statement,
                            field="credit_cards_list",
                            value=", ".join(card_names[:8]),
                            source_url=src_url,
                        )
                        claims.append(c)
                        answer_parts.append(statement)

            # 2. Process tables (fees, limits, installments)
            for t in extracted.get("tables", []):
                t_name = t.get("table_name", "Table")
                for rec in t.get("records", []):
                    for k, v in rec.items():
                        if k and v and k != v and k.lower() not in ("details", "col_0", "col_1"):
                            statement = f"{t_name}: {k} is {v}."
                            if statement not in seen_statements:
                                seen_statements.add(statement)
                                c = Claim(
                                    id=f"c_{uuid.uuid4().hex[:8]}",
                                    statement=statement,
                                    field=k,
                                    value=v,
                                    source_url=src_url,
                                )
                                claims.append(c)

            # 3. Process PDF tables
            for pt in extracted.get("pdf_tables", []):
                t_name = pt.get("table_name", "PDF Fee Table")
                for rec in pt.get("records", []):
                    for k, v in rec.items():
                        if k and v and len(str(v)) > 0:
                            statement = f"{t_name}: {k} is {v}."
                            if statement not in seen_statements:
                                seen_statements.add(statement)
                                c = Claim(
                                    id=f"c_{uuid.uuid4().hex[:8]}",
                                    statement=statement,
                                    field=k,
                                    value=str(v),
                                    source_url=src_url,
                                )
                                claims.append(c)

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
