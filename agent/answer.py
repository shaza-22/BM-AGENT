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
# The same defensive parse the planner and link selector use: models wrap JSON
# in prose, and one place to fix that is better than three.
from agent.planner import _extract_json
from agent.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

SYSTEM = (
    "You write short, factual answers for a bank's research assistant. "
    "You are given a numbered list of facts that were extracted from pages the "
    "assistant actually fetched, and nothing else is true as far as you are "
    "concerned. Every figure you write must be copied from those facts."
)

# One line both composition prompts carry, so anything that needs to recognise
# "this is the composing call" -- a test double, a recording harness -- has a
# single stable string to match rather than one per prompt variant.
COMPOSING_MARKER = "Write the answer to the user's question"

TIERS = ("fact", "analysis", "judgment")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tier": {"type": "string", "enum": list(TIERS)},
                    "text": {"type": "string"},
                },
                "required": ["tier", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sentences"],
    "additionalProperties": False,
}

TIERED_PROMPT = """\
The user asked:
{task}

These are the facts this run verified. Each is a label, its value, and the page it came from.

{facts}

Write the answer to the user's question from these facts, as a list of
sentences, each tagged with what kind of statement it is.

**fact** — restated from a page. Copy every figure exactly as written above.
  Group related facts under short bold headings and put one fact per line as
  "Label — value". Open with a short line saying what follows and close with one
  saying where the figures came from; tag those as fact too.

**analysis** — something you worked out *from* the facts: a comparison, a total,
  a pattern, a summary. This is the part that makes the answer research rather
  than a list, so include it whenever the facts support it. Every figure and
  label you use must still come from the list above; only the inference is
  yours. For example: two fees being equal means no increase after the first
  year — that conclusion is not written on any page, but both figures are.

{judgment_rule}

Rules that hold for every tier:
- Never write a figure that is not in the list above, copied exactly, including
  the currency and any % sign.
- Never pair a label with a different fact's value.
- Do not add background, history, or anything not derived from the list.
- If the facts do not answer the question, say which part is missing.
- Write in {language_name}.
- Keep it short enough to read at a glance.
"""

JUDGMENT_ALLOWED = """\
**judgment** — a recommendation or a verdict, where the question asks for one.
  The premises must be facts from the list; the opinion is yours, and you should
  say plainly that it is a suggestion. Name the trade-off, not only the winner.
"""

JUDGMENT_WITHHELD = """\
Do not offer a recommendation, a verdict, or an opinion about which option is
better. The user asked what is true, not what to choose.
"""

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
    # Sentences by tier, each already through grounding. Kept separate from
    # ``text`` because the interface has to show which statements the bank's
    # website is responsible for and which the agent is -- a derived conclusion
    # must never wear a verified fact's badge.
    tiers: dict[str, list[str]] = field(default_factory=dict)
    tiered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "used": self.used,
            "reason": self.reason,
            "claims_offered": self.claims_offered,
            "grounding": self.report.to_dict() if self.report else None,
            "tiered": self.tiered,
            "tiers": {tier: list(lines) for tier, lines in self.tiers.items()},
        }


def _humanise_label(label: str) -> str:
    """Never put an internal identifier in front of the model.

    Facts are labelled by ``entity``, falling back to ``field`` -- and ``field``
    holds programmatic names like ``entity_list``. The model has no way to know
    that is not what the bank calls the thing, so it copies it into the answer
    verbatim, and the reader is shown a variable name. Anything that looks like
    an identifier is turned back into words here.
    """
    text = (label or "").strip()
    if "_" in text and " " not in text:
        text = text.replace("_", " ").strip()
        return text[:1].upper() + text[1:]
    return text


def _facts_block(claims: list[dict], limit: int) -> str:
    lines = []
    for index, claim in enumerate(claims[:limit], 1):
        label = _humanise_label(claim.get("entity") or claim.get("field") or "")
        value = claim.get("value") or ""
        source = claim.get("source_url") or ""
        if label and value:
            lines.append(f"{index}. {label}: {value}   [{source}]")
        else:
            lines.append(f"{index}. {claim.get('statement', '')}   [{source}]")
    return "\n".join(lines)


def _wants_judgment(task: str, task_type: str | None) -> bool:
    """Is this a question that asks to be advised, rather than told?

    Driven by the plan's own task type -- the same classification the expansion
    gate uses -- so there is one notion of "this asks for a recommendation" in
    the system rather than two that can disagree.
    """
    return (task_type or "").lower() in ("recommendation", "comparison")


def _parse_tiers(raw: str) -> list[tuple[str, str]]:
    """(tier, text) pairs from a tiered reply, dropping anything unusable."""
    parsed = _extract_json(raw)
    items = parsed.get("sentences")
    if not isinstance(items, list):
        raise ValueError("reply has no sentences list")
    out: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        tier = str(item.get("tier") or "fact").strip().lower()
        if not text:
            continue
        # An unrecognised tier is treated as the strictest one rather than
        # dropped: the sentence still has to pass grounding, and calling it a
        # fact means the interface will not present it as the agent's opinion.
        out.append((tier if tier in TIERS else "fact", text))
    if not out:
        raise ValueError("no usable sentences in the reply")
    return out


def compose_tiered(
    task: str,
    claims: list[dict],
    llm: LLMClient,
    *,
    language: str | None = None,
    task_type: str | None = None,
    max_facts: int = config.COMPOSE_MAX_FACTS,
) -> Composition:
    """Write the answer as tagged sentences: fact, analysis, and maybe judgment.

    One model call, the same one composition already spent -- the tiers are a
    shape the reply takes, not extra work. Grounding then runs per sentence,
    **unchanged**: every figure must be quoted, every sentence anchored, and a
    label must keep its own value. Analysis and judgment are subject to exactly
    those rules; what is theirs is the inference between the figures, never the
    figures.
    """
    if not claims:
        return Composition(
            attempted=False,
            reason="no verified claims; refusing to generate without evidence to ground on",
        )

    prompt = TIERED_PROMPT.format(
        task=task,
        facts=_facts_block(claims, max_facts),
        judgment_rule=(JUDGMENT_ALLOWED if _wants_judgment(task, task_type)
                       else JUDGMENT_WITHHELD),
        language_name=LANGUAGE_NAMES.get(language or "en", "English"),
    )

    composition = Composition(attempted=True, tiered=True,
                              claims_offered=min(len(claims), max_facts))
    try:
        raw = llm.complete(prompt, system=SYSTEM, schema=ANSWER_SCHEMA)
    except LLMError as exc:
        logger.warning("tiered composition failed, keeping the template answer: %s", exc)
        composition.reason = f"model unavailable: {exc}"
        return composition

    try:
        sentences = _parse_tiers(raw)
    except Exception as exc:
        logger.warning("tiered composition unparseable, keeping the template answer: %s", exc)
        composition.reason = f"unparseable reply: {exc}"
        return composition

    # Whether an opinion is wanted is decided here, not by the prompt. The
    # prompt asks; a guard clause guarantees. A model handed a fee question
    # will sometimes volunteer a recommendation anyway, and an unasked-for
    # opinion from a bank's assistant is the one kind of sentence worth
    # dropping outright rather than relabelling -- calling it analysis would
    # only hide it behind the wrong badge.
    allow_judgment = _wants_judgment(task, task_type)

    kept: dict[str, list[str]] = {tier: [] for tier in TIERS}
    struck: list[tuple[str, str]] = []
    dropped_judgments = 0
    total = 0
    for tier, text in sentences:
        total += 1
        if tier == "judgment" and not allow_judgment:
            dropped_judgments += 1
            continue
        report = check_grounding(text, claims[:max_facts])
        if report.text.strip():
            kept[tier].append(report.text.strip())
        struck.extend(report.struck)

    composition.tiers = {tier: lines for tier, lines in kept.items() if lines}
    composition.report = GroundingReport(
        text="", kept=[line for lines in kept.values() for line in lines], struck=struck,
    )

    if dropped_judgments:
        logger.info(
            "dropped %d unasked-for recommendation sentence(s): the task is %r, "
            "which does not ask to be advised", dropped_judgments, task_type or "general",
        )

    if not composition.tiers:
        composition.reason = "every generated sentence failed grounding"
        logger.warning("tiered composition discarded: all %d sentence(s) struck", total)
        return composition

    # The rendered text keeps facts first, then the derived tiers, so the
    # answer reads in the order the reasoning happened.
    blocks = ["\n".join(composition.tiers[tier]) for tier in TIERS if tier in composition.tiers]
    composition.text = "\n\n".join(blocks)
    composition.used = True
    composition.reason = (
        f"{len(composition.report.kept)}/{total} sentences grounded "
        f"({', '.join(f'{len(v)} {k}' for k, v in composition.tiers.items())})"
    )
    logger.info("answer composed in tiers: %s", composition.reason)
    return composition


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
