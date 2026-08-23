"""
Decomposing a task into sub-goals with the model, or falling back to keywords.

What it does
    Asks the model to break the user's request into the smallest set of
    sub-goals that would answer it, then rewrites the vendored planner's
    ``Plan`` in place so the rest of the loop is unchanged.

Inputs / Outputs
    ``plan_with_model(task, plan, llm, language=...)`` -> :class:`PlanDraft`,
    and the ``Plan`` is mutated only when the draft is usable.

Why it is needed
    ``person_b.planning.planner.plan_task`` decomposes by keyword: it matches
    "credit card", "loan", "account" and otherwise emits a single sub-goal that
    restates the task. Measured across eight varied tasks, it produced exactly
    one sub-goal every time -- so a plan panel showed one restated line, and
    "planning" was a word for string interpolation.

    That deterministic planner is still what runs when this is off or fails,
    and it is still what decides ``task_type`` and ``target_fields``. This
    replaces the *questions*, not the machinery around them.

Cost, and why the cap is three
    The model is called once per hop, measured. With ``LOOP_MAX_LLM_CALLS = 12``
    and one call already reserved for composing the answer, planning takes one
    more and leaves ten for navigation. Pages on this site resolve in one to
    three hops, so three sub-goals fit (1 + 9 + 1 = 11) and four do not. When a
    sub-goal runs deep and the budget is spent, the remainder are marked
    not-available with the reason and appear in the answer -- the failure is
    visible, which is the same way every other budget here degrades.

Failing safe
    Every failure path keeps the deterministic plan: an unreachable model, an
    unparseable reply, an empty list, sub-goals that are blank or duplicated.
    Planning can improve a plan; it can never leave a run without one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agent import config
from agent.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["question", "why"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["sub_goals", "reasoning"],
    "additionalProperties": False,
}

SYSTEM = (
    "You plan research on a bank's public website. You break a request into the "
    "smallest number of sub-goals that would answer it, where each sub-goal is "
    "something a person could look up on one page of the site. You never answer "
    "the request yourself and you never assume what the site contains."
)

PROMPT = """\
A user asked a bank's research assistant:

{task}

Break this into the sub-goals needed to answer it. An agent will navigate the
bank's website from its homepage for each one, separately, and each sub-goal
should be answerable from a single page.

Rules:
- Use as few sub-goals as the request needs. One is correct and expected when
  the request asks for one thing — do not split a simple lookup to look busy.
- At most {max_sub_goals}. Every extra sub-goal costs a full walk of the site.
- Each must be a self-contained question. The agent navigating it sees only
  that question, not the original request and not the other sub-goals.
- Only name a product if the user named it. You do not know what the site
  lists, so "find the fees for each card" is not a sub-goal you can write —
  "find the list of credit cards" is, and the rest follows from what it finds.
- Do not add sub-goals for background, history, or context nobody asked for.
- Write the questions in {language_name}.

Return the sub-goals in the order they should be researched, each with one
short line saying why it is needed.
"""

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass
class PlanDraft:
    """What planning produced, and whether it was used."""

    sub_goals: list[dict[str, str]] = field(default_factory=list)
    reasoning: str = ""
    attempted: bool = False
    used: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "used": self.used,
            "reason": self.reason,
            "reasoning": self.reasoning,
            "sub_goals": list(self.sub_goals),
        }


def _extract_json(raw: str) -> dict:
    """Same defensive parse the link selector uses: models wrap JSON in prose."""
    text = _FENCE_RE.sub("", (raw or "").strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected a JSON object", text, 0)
    return parsed


def parse_plan(raw: str, *, max_sub_goals: int) -> list[dict[str, str]]:
    """Pull clean sub-goals out of a reply, dropping anything unusable.

    Deliberately forgiving about shape and strict about content: a truncated
    reply that still yielded two good questions is worth more than nothing,
    but a blank or duplicated question is not a sub-goal and would cost a full
    navigation to discover that.
    """
    parsed = _extract_json(raw)
    items = parsed.get("sub_goals")
    if not isinstance(items, list):
        raise ValueError("reply has no sub_goals list")

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            item = {"question": item, "why": ""}
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if len(question) < 8:            # not a question anyone can navigate
            continue
        key = re.sub(r"\W+", " ", question.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"question": question, "why": str(item.get("why") or "").strip()})
        if len(cleaned) >= max_sub_goals:
            break
    if not cleaned:
        raise ValueError("no usable sub-goals in the reply")
    return cleaned


def plan_with_model(
    task: str,
    plan: Any,
    llm: LLMClient,
    *,
    language: str | None = None,
    max_sub_goals: int = config.MAX_PLANNED_SUB_GOALS,
) -> PlanDraft:
    """Replace a deterministic plan's sub-goals with model-authored ones.

    ``plan`` is mutated in place and only on success, so a caller that ignores
    the return value still holds a valid plan. ``task_type`` and
    ``target_fields`` are left exactly as the vendored planner set them: they
    drive the validator's own branches, and changing what feeds those from here
    would be reaching into the other half of the project.
    """
    draft = PlanDraft()
    if not config.LLM_PLANNING:
        draft.reason = "LLM planning is off; using the keyword planner"
        return draft

    draft.attempted = True
    prompt = PROMPT.format(
        task=task,
        max_sub_goals=max_sub_goals,
        language_name=LANGUAGE_NAMES.get(language or "en", "English"),
    )

    try:
        raw = llm.complete(prompt, system=SYSTEM, schema=PLAN_SCHEMA)
    except LLMError as exc:
        # A plan already exists. Failing here must cost the wording of the
        # sub-goals, never the run.
        logger.warning("planning failed, keeping the keyword plan: %s", exc)
        draft.reason = f"model unavailable: {exc}"
        return draft

    try:
        draft.sub_goals = parse_plan(raw, max_sub_goals=max_sub_goals)
    except Exception as exc:
        logger.warning("could not parse the plan, keeping the keyword plan: %s", exc)
        draft.reason = f"unparseable plan: {exc}"
        return draft

    try:
        draft.reasoning = str(_extract_json(raw).get("reasoning") or "").strip()
    except Exception:
        draft.reasoning = ""

    _apply(plan, draft.sub_goals)
    draft.used = True
    draft.reason = f"decomposed into {len(draft.sub_goals)} sub-goal(s)"
    logger.info(
        "planned: task=%r -> %d sub-goal(s): %s (%s)",
        task, len(draft.sub_goals), [sg["question"] for sg in draft.sub_goals],
        draft.reasoning or "no reasoning given",
    )
    return draft


def _apply(plan: Any, sub_goals: list[dict[str, str]]) -> None:
    """Rewrite the plan's sub-goals, keeping everything else the planner set."""
    from person_b.models import SubGoal, SubGoalStatus

    template = plan.sub_goals[0] if plan.sub_goals else None
    target_fields = list(getattr(template, "target_fields", []) or [])

    plan.sub_goals = [
        SubGoal(
            id=f"sg_{index:03d}",
            question=item["question"],
            status=SubGoalStatus.PENDING,
            target_fields=list(target_fields),
            created_from="llm_plan",
            metadata={"why": item.get("why", "")},
        )
        for index, item in enumerate(sub_goals, 1)
    ]
    plan.version += 1
