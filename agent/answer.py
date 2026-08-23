"""
Composing the final answer: prose from verified facts, or nothing.

What it does
    Asks the model to write the answer to the user's question using *only* the
    claims the run verified, then puts the result through
    :mod:`agent.grounding` before anyone sees it.

Inputs / Outputs
    ``compose_answer(task, claims, llm, language=...)`` ->
    :class:`Composition`: the grounded prose, the grounding report, and whether
    composition happened at all.

Why it is needed
    The vendored synthesis is template-based with no model in it, by design --
    zero quota, no hallucination risk. The cost is that it can fill slots and
    nothing else: it produces "Fees and charges: Issuance — EGP 250." per row
    and cannot say which of those rows answers the question. This step buys
    prose while keeping the grounding, and falls back to the template whenever
    it cannot.

The hard rule
    **No claims, no call.** If the run verified nothing, the model is not
    invited to write anything -- it would compose from what it happens to know
    about the bank, and every word of that is invention wearing the interface's
    credibility. This is the single largest hallucination risk in generating an
    answer at all, so it is a guard clause at the top of the function rather
    than a prompt instruction, which a model may ignore.

    It also covers Arabic for free: the vendored extraction cannot read Arabic
    pages, so an Arabic run produces no claims, so nothing is generated and the
    template "not found" answer stands. That is the visible failure the other
    author asked for, not a compensating guess.

What still gets through
    Everything :mod:`agent.grounding` says it cannot catch, chiefly
    recombination of real figures. Read that module's docstring before
    describing this as verified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent import config
from agent.grounding import GroundingReport, check_grounding
from agent.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

SYSTEM = (
    "You write short, factual answers for a bank's research assistant. "
    "You are given a numbered list of facts that were extracted from pages the "
    "assistant actually fetched, and nothing else is true as far as you are "
    "concerned. Every figure you write must be copied from those facts."
)

PROMPT = """\
The user asked:
{task}

These are the facts this run verified. Each is a label, its value, and the page it came from.

{facts}

Write the answer to the user's question using only these facts.

Rules:
- Use only figures that appear above, copied exactly as written (including the currency and any % sign).
- Keep each fact with its own label. Do not pair a label with a different fact's value.
- Do not add background, history, advice, or anything not in the list above.
- If the facts do not answer the question, say exactly which part is missing instead of filling it in.
- Answer in {language_name}.

Shape:
- Open with one short line saying what follows.
- Group related facts together under short bold headings, e.g. **Card fees**,
  **Charges and penalties**, **Interest and installments**. Choose the groupings
  from the facts themselves.
- Put one fact per line as "Label — value".
- Leave a blank line between groups.
- Close with one short line saying where the figures came from.
- Do not repeat a fact in more than one group, and keep the whole thing short
  enough to read at a glance.
"""

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}


@dataclass
class Composition:
    """The outcome of trying to write a grounded answer."""

    text: str = ""
    attempted: bool = False
    used: bool = False
    reason: str = ""
    report: GroundingReport | None = None
    claims_offered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "used": self.used,
            "reason": self.reason,
            "claims_offered": self.claims_offered,
            "grounding": self.report.to_dict() if self.report else None,
        }


def _facts_block(claims: list[dict], limit: int) -> str:
    lines = []
    for index, claim in enumerate(claims[:limit], 1):
        label = claim.get("entity") or claim.get("field") or ""
        value = claim.get("value") or ""
        source = claim.get("source_url") or ""
        if label and value:
            lines.append(f"{index}. {label}: {value}   [{source}]")
        else:
            lines.append(f"{index}. {claim.get('statement', '')}   [{source}]")
    return "\n".join(lines)


def compose_answer(
    task: str,
    claims: list[dict],
    llm: LLMClient,
    *,
    language: str | None = None,
    max_facts: int = config.COMPOSE_MAX_FACTS,
) -> Composition:
    """Write the answer from the verified claims, or decline and say why."""
    if not claims:
        # The hard gate. See the module docstring: this is not an optimisation.
        return Composition(
            attempted=False,
            reason="no verified claims; refusing to generate without evidence to ground on",
        )

    facts = _facts_block(claims, max_facts)
    prompt = PROMPT.format(
        task=task,
        facts=facts,
        language_name=LANGUAGE_NAMES.get(language or "en", "English"),
    )

    try:
        raw = llm.complete(prompt, system=SYSTEM)
    except LLMError as exc:
        # Composition is a finishing touch on an answer that already exists in
        # template form. It must never be the reason a run reports failure.
        logger.warning("answer composition failed, keeping the template answer: %s", exc)
        return Composition(attempted=True, reason=f"model unavailable: {exc}",
                           claims_offered=min(len(claims), max_facts))

    report = check_grounding(raw or "", claims[:max_facts])
    composition = Composition(
        text=report.text,
        attempted=True,
        report=report,
        claims_offered=min(len(claims), max_facts),
    )

    if not report.text.strip():
        composition.reason = "every generated sentence failed grounding"
        logger.warning(
            "composition discarded: all %d sentence(s) struck", report.sentences_total
        )
        return composition

    composition.used = True
    composition.reason = (
        f"{len(report.kept)}/{report.sentences_total} generated sentences grounded"
    )
    logger.info("answer composed: %s", composition.reason)
    return composition
