"""
The orchestrator: one user task, a plan, several navigations, one answer.

What it does
    Drives the vendored intelligence layer (``src/person_b``) across this
    layer's navigator. Plans the task into sub-goals, navigates each one live
    from the seed, feeds every fetched page to their validator, expands the
    plan from what was discovered, then synthesises, verifies and finalises.

Inputs
    ``ResearchLoop(llm, fetcher=..., on_event=...).run(task)``.

Outputs
    A :class:`LoopResult`: the plan with each sub-goal's outcome, the final
    verified answer, the support rate, and every source actually fetched.

Why it is needed
    The two halves meet in exactly one place, and this is it. Their
    ``validate()`` takes a sub-goal and page text; the navigator wants
    ``validate_fn(sub_goal, page) -> dict``. Everything else here is budget and
    bookkeeping.

The anti-hallucination property, and how it is preserved
    Nothing enters the loop that this run did not fetch. Page text comes only
    from ``NavigationResult`` trails, ``visited_urls`` is built from pages that
    actually returned content, and that same list is what
    ``validate_answer`` checks claims against. There is no fixture loader, no
    cache priming and no index on this path -- a claim citing a page the run
    never fetched cannot survive, because the URL is not in the list.

Budgets
    The per-sub-goal caps (``MAX_HOPS``, ``MAX_PAGES``) do not bound a plan;
    they multiply. Measured: one credit-cards page expands to 13 sub-goals,
    which at 15 pages each is ~180 requests to a WAF-protected site and 13+
    model calls against a 20/day free tier. The budgets that hold the line are
    the global ones -- ``LOOP_MAX_PAGES`` and ``LOOP_MAX_LLM_CALLS`` -- spent
    across the whole plan rather than reset per sub-goal. When one runs out,
    remaining sub-goals are marked not-available with the reason, so the answer
    says what it could not look at instead of quietly omitting it.

Partial verdicts do not stop navigation
    ``partial`` means the entity is relevant and some requested fields were
    found -- precisely the state where the rest is one hop deeper. Stopping
    wastes the remaining hops; discarding the evidence wastes work already
    paid for. So the loop keeps going and keeps the partial: if nothing
    resolves, the answer is synthesised from accumulated partials with the gaps
    named, which beats reporting a flat failure. See
    ``config.PARTIAL_STOPS_NAVIGATION``.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from agent import config
from agent.acceptance import AcceptanceGate, GateDecision, gate_from_config
from agent.answer import compose_answer, compose_tiered
from agent.extraction import Extraction, extract_facts
from agent.planner import PlanDraft, plan_with_model
from agent.llm import LLMClient, LLMError
from agent.navigator import NavigationResult, Navigator
from agent.trail_log import StepLogger
from browsing.fetcher import Fetcher, PageDict

logger = logging.getLogger(__name__)

EventFn = Callable[[str, dict[str, Any]], None]


# --------------------------------------------------------------------------
# The seam: their validate() behind the navigator's validate_fn
# --------------------------------------------------------------------------
@dataclass
class SubGoalOutcome:
    """What one sub-goal's navigation produced."""

    sub_goal_id: str
    question: str
    status: str  # resolved | partial | not_available | unreadable | unresolved
    reason: str = ""
    source_url: str | None = None
    verdict: dict[str, Any] | None = None
    nav_status: str = ""
    # Set when the model read the page after keyword extraction found nothing.
    extraction: dict[str, Any] | None = None
    hops_used: int = 0
    pages_fetched: int = 0
    gate: GateDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_goal_id": self.sub_goal_id,
            "question": self.question,
            "status": self.status,
            "reason": self.reason,
            "source_url": self.source_url,
            "nav_status": self.nav_status,
            "hops_used": self.hops_used,
            "pages_fetched": self.pages_fetched,
            "gate": self.gate.to_dict() if self.gate else None,
            "extraction": self.extraction,
        }


@dataclass
class LoopResult:
    """The whole run: what was planned, what was found, what survived checking."""

    task: str
    plan_id: str
    task_type: str
    outcomes: list[SubGoalOutcome] = field(default_factory=list)
    answer: str = ""
    source_urls: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    support_rate: float = 1.0
    claims_total: int = 0
    claims_supported: int = 0
    visited_urls: list[str] = field(default_factory=list)
    pages_used: int = 0
    llm_calls_used: int = 0
    budget_exhausted: str | None = None
    gate_rejections: int = 0
    elapsed_s: float = 0.0
    # How the answer text was produced, and what grounding did to it. Kept so
    # the interface can say which it is showing rather than implying prose and
    # template carry the same warranty.
    answer_source: str = "template"          # "template" | "composed"
    composition: dict[str, Any] | None = None
    template_answer: str = ""
    # How the plan was decomposed, and why. "keyword" is the vendored planner;
    # "model" is agent/planner.py. Reported so the interface can say which,
    # rather than presenting a keyword match as agentic planning.
    answer_tiers: dict[str, list[str]] = field(default_factory=dict)
    plan_source: str = "keyword"             # "keyword" | "model"
    plan_reasoning: str = ""
    # Every hop of every sub-goal, in order. Kept so the session can summarise
    # a turn without the loop having to hand back page content.
    trail: list[Any] = field(default_factory=list)
    # Set when a navigation hit the WAF. A block is not a result to synthesise
    # from -- it aborts the whole task, exactly as it does for one navigation.
    blocked_reason: str | None = None
    # Set when the model could not be reached after llm.py's own retries. This
    # must not be reported as "nothing found": a spent daily quota and a
    # genuinely unanswerable question look identical in the answer text and
    # completely different to whoever has to act on it.
    error_reason: str | None = None

    @property
    def resolved_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "resolved")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "plan_id": self.plan_id,
            "task_type": self.task_type,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "answer": self.answer,
            "source_urls": list(self.source_urls),
            "not_found": list(self.not_found),
            "support_rate": self.support_rate,
            "claims_total": self.claims_total,
            "claims_supported": self.claims_supported,
            "visited_urls": list(self.visited_urls),
            "pages_used": self.pages_used,
            "llm_calls_used": self.llm_calls_used,
            "budget_exhausted": self.budget_exhausted,
            "blocked_reason": self.blocked_reason,
            "error_reason": self.error_reason,
            "gate_rejections": self.gate_rejections,
            "answer_source": self.answer_source,
            "plan_source": self.plan_source,
            "plan_reasoning": self.plan_reasoning,
            "composition": self.composition,
            "template_answer": self.template_answer,
            "resolved_count": self.resolved_count,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _entities_named_by(task: str, names: list[str]) -> list[str]:
    """Which of these entity names does the task actually single out?

    The discriminating words are derived from the candidates themselves rather
    than from a list: any word shared by *every* discovered entity says nothing
    about which one is meant. Given eight cards, "credit" and "card" are shared
    and drop out, leaving "classic", "gold", "titanium" and so on -- so
    "Compare the Classic and Gold credit cards" singles out two of the eight.

    Deriving the stop set per run is what keeps this topic-agnostic: it holds
    identically for loans, accounts or anything else the site lists, and names
    no category itself.
    """
    if len(names) < 2:
        return []
    per_name = {name: _words(name) for name in names}

    # A word carries no discriminating power when most of the candidates have
    # it. Not *all* of them: one outlier name ("Visa Infinite") that happens to
    # omit the common noun would otherwise keep "credit" and "card" in play,
    # and then every card matches a task mentioning cards at all. Majority
    # presence is the test, so a single outlier cannot defeat it.
    frequency: dict[str, int] = {}
    for words in per_name.values():
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1
    threshold = len(per_name) / 2
    shared = {word for word, count in frequency.items() if count > threshold}

    task_words = _words(task)
    return [
        name for name, words in per_name.items()
        if (words - shared) & task_words
    ]


_VERDICT_RANK = {"resolved": 3, "partial": 2, "unreadable": 1, "unresolved": 0}


class _Verdicts:
    """Every verdict one sub-goal's navigation produced, keyed by page.

    Needed because ``NavigationResult.extracted`` carries the *extracted data*
    of the resolving verdict, not the verdict itself -- the reason, status and
    gate decision are not on it. Rather than widen the navigator's result shape
    for the benefit of one caller, the loop keeps what it hands out.

    It also keeps the best non-resolving verdict, which is what makes the
    partial policy possible: a run that ends ``exhausted`` may still have seen
    a page with half the answer on it, and that is worth more than nothing.
    """

    def __init__(self) -> None:
        self.by_url: dict[str, dict[str, Any]] = {}
        self.best: dict[str, Any] | None = None

    def record(self, url: str, verdict: dict[str, Any]) -> None:
        self.by_url[url] = verdict
        rank = _VERDICT_RANK.get(str(verdict.get("status", "unresolved")), 0)
        if verdict.get("resolved"):
            rank = _VERDICT_RANK["resolved"]
        best_rank = -1
        if self.best is not None:
            best_rank = _VERDICT_RANK.get(str(self.best.get("status", "unresolved")), 0)
            if self.best.get("resolved"):
                best_rank = _VERDICT_RANK["resolved"]
        if rank > best_rank:
            self.best = verdict


class _CountingLLM:
    """Wraps an LLMClient to count calls against the plan's budget.

    A wrapper rather than a counter inside the navigator because the budget
    belongs to the plan: several navigations share it, and the navigator has
    no reason to know a plan exists.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls = 0

    def complete(self, prompt: str, *, system: str | None = None, schema: Any = None) -> str:
        self.calls += 1
        return self._inner.complete(prompt, system=system, schema=schema)


class ResearchLoop:
    """Runs one task end to end across Person B's plan."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        fetcher: Fetcher | None = None,
        gate: AcceptanceGate | None = None,
        on_event: EventFn | None = None,
        step_logger: StepLogger | None = None,
        language: str | None = None,
        max_sub_goals: int = config.MAX_SUB_GOALS,
        max_pages: int = config.LOOP_MAX_PAGES,
        max_llm_calls: int = config.LOOP_MAX_LLM_CALLS,
        max_expansion_depth: int = config.MAX_EXPANSION_DEPTH,
        compose: bool | None = None,
        planning: bool | None = None,
        fallback: bool | None = None,
        tiers: bool | None = None,
    ) -> None:
        self._llm = _CountingLLM(llm)
        self._fetcher = fetcher
        self._gate = gate if gate is not None else gate_from_config()
        self._on_event = on_event or (lambda name, data: None)
        self._step_logger = step_logger
        self._language = language
        self._max_sub_goals = max_sub_goals
        self._max_pages = max_pages
        self._max_llm_calls = max_llm_calls
        self._max_expansion_depth = max_expansion_depth
        self._compose = config.COMPOSE_ANSWER if compose is None else compose
        self._fallback = (config.LLM_EXTRACTION_FALLBACK if fallback is None else fallback)
        self._tiers = config.ANSWER_TIERS if tiers is None else tiers
        self._planning = config.LLM_PLANNING if planning is None else planning

        self._pages_used = 0
        self._planned_sub_goals = 0
        self._page_text: dict[str, str] = {}
        self._visited: list[str] = []
        self._gate_rejections = 0

    # -- events ------------------------------------------------------------
    def _emit(self, name: str, **data: Any) -> None:
        try:
            self._on_event(name, data)
        except Exception:  # a subscriber must never break a run
            logger.exception("on_event raised for %s", name)

    def _budget(self) -> dict[str, Any]:
        return {
            "pages_used": self._pages_used,
            "pages_max": self._max_pages,
            "llm_calls_used": self._llm.calls,
            "llm_calls_max": self._max_llm_calls,
        }

    def _exhausted(self) -> str | None:
        if self._pages_used >= self._max_pages:
            return f"page budget spent ({self._pages_used}/{self._max_pages})"
        # Navigation stops one call early when composition is on, so the run
        # always has a call left to write the answer with. Spending the last
        # call on a hop and then having no budget to phrase the result is the
        # wrong trade: the hop might find nothing, the answer is certain.
        navigation_budget = self._max_llm_calls - (1 if self._compose else 0)
        if self._llm.calls >= navigation_budget:
            return f"model-call budget spent ({self._llm.calls}/{navigation_budget})"
        return None

    # -- the validate_fn seam ---------------------------------------------
    def _make_validate_fn(
        self, sub_goal: Any, collected: "_Verdicts"
    ) -> Callable[[str, PageDict], dict]:
        """Adapt Person B's ``validate()`` to the navigator's contract.

        Their signature is four-valued (``resolved`` / ``partial`` /
        ``unresolved`` / ``unreadable``) and the navigator wants a boolean, so
        the mapping is: only ``resolved`` stops navigation. The full verdict is
        kept alongside so nothing is lost by the narrowing.

        This is also where the acceptance gate runs, because it is the one
        place that sees a verdict and its page together.
        """
        from person_b.api import validate as person_b_validate

        def validate_fn(_sub_goal_text: str, page: PageDict) -> dict:
            url = page.get("url") or ""
            text = page.get("text") or ""
            self._page_text[url] = text
            self._gate.observe(url, text)

            is_pdf = "pdf" in (page.get("content_type") or "")
            verdict = person_b_validate(
                sub_goal,
                text,
                source_url=url,
                content_type="pdf" if is_pdf else "text",
                status_code=page.get("status"),
            )

            decision = self._gate.judge(verdict, url, text)
            verdict["gate"] = decision.to_dict()
            if not decision.accepted:
                self._gate_rejections += 1
                # Logged at INFO on purpose: a withheld resolve changes what
                # the run does, and a silent override is exactly the kind of
                # thing that makes a system impossible to debug from outside.
                logger.info(
                    "acceptance gate withheld a resolve: url=%s sub_goal=%r "
                    "validator_reason=%r gate=%s",
                    url, getattr(sub_goal, "question", sub_goal),
                    verdict.get("reason"), decision.reason,
                )
                self._emit(
                    "gate",
                    url=url,
                    sub_goal_id=getattr(sub_goal, "id", None),
                    validator_reason=verdict.get("reason", ""),
                    **decision.to_dict(),
                )
                verdict = {
                    **verdict,
                    "resolved": False,
                    "status": "unresolved",
                    "reason": (
                        f"withheld by acceptance gate: {decision.reason} "
                        f"(validator said: {verdict.get('reason', '')})"
                    ),
                }
            collected.record(url, verdict)
            return verdict

        return validate_fn

    # -- the run -----------------------------------------------------------
    def run(self, task: str) -> LoopResult:
        from person_b.api import (
            apply_validation,
            expand_plan,
            finalize,
            next_pending_sub_goal,
            plan_task,
            synthesize,
            validate_answer,
        )
        from person_b.config import PersonBConfig, default_config

        started = time.monotonic()

        # Their expansion cap is overridden here rather than edited in their
        # config, so a future drop from them re-vendors cleanly.
        pb_config = PersonBConfig(
            **{**default_config.to_dict(),
               "reasoning_api_key": None,
               "max_expansion_sub_goals": self._max_sub_goals}
        )

        # The keyword planner runs first and always. It sets task_type and
        # target_fields, which the validator's own branches read, and it is the
        # plan that stands if model planning is off or fails.
        plan = plan_task(task, pb_config)
        result = LoopResult(task=task, plan_id=plan.id, task_type=plan.goal.task_type)

        draft = (plan_with_model(task, plan, self._llm, language=self._language)
                 if self._planning else PlanDraft(reason="planning disabled for this run"))
        if draft.used:
            result.plan_source = "model"
            result.plan_reasoning = draft.reasoning
        self._planned_sub_goals = len(plan.sub_goals) if draft.used else 0
        self._emit(
            "plan",
            plan_id=plan.id,
            task_type=plan.goal.task_type,
            target_fields=plan.goal.metadata.get("target_fields", []),
            sub_goals=[{"id": sg.id, "question": sg.question, "status": sg.status.value,
                        "why": sg.metadata.get("why", "")}
                       for sg in plan.sub_goals],
            gate_enabled=self._gate.enabled,
            plan_source=result.plan_source,
            plan_reasoning=result.plan_reasoning,
            **self._budget(),
        )

        validated_results: list[dict[str, Any]] = []
        attempted: set[str] = set()

        while True:
            sub_goal = next_pending_sub_goal(plan)
            if sub_goal is None:
                break
            if sub_goal.id in attempted:
                # Should be unreachable: every attempt below leaves the
                # sub-goal in a terminal state. Kept as a guard because the
                # alternative to a wrong answer here is an infinite loop.
                logger.warning("sub-goal %s came back PENDING after an attempt", sub_goal.id)
                from person_b.api import mark_not_available

                mark_not_available(plan, sub_goal, "attempted but left pending")
                continue
            attempted.add(sub_goal.id)

            spent = self._exhausted()
            if spent is not None:
                result.budget_exhausted = spent
                self._mark_remaining_unavailable(plan, spent, result)
                break

            outcome = self._run_sub_goal(plan, sub_goal, result)
            result.outcomes.append(outcome)
            halt = result.blocked_reason or result.error_reason
            if halt is not None:
                # A block: continuing would keep knocking on a WAF that has
                # already flagged this run, which is how a block becomes a ban.
                # An error: llm.py has already exhausted its retries, so the
                # next sub-goal would fail the same way and cost a page doing
                # it.
                self._mark_remaining_unavailable(plan, halt, result)
                break
            from person_b.api import mark_not_available
            from person_b.models import SubGoalStatus

            if outcome.verdict is not None:
                apply_validation(plan, sub_goal, outcome.verdict)
                if outcome.status in ("resolved", "partial"):
                    validated_results.append(outcome.verdict)

            # apply_validation returns an *unresolved* sub-goal to PENDING,
            # which is right for a validator being handed page after page but
            # wrong once a whole navigation has been spent on it: the next
            # iteration would pick the same sub-goal again, and guarding that
            # by stopping meant the sub-goals behind it were dropped without a
            # word. A spent sub-goal is not-available, and says so.
            if sub_goal.status == SubGoalStatus.PENDING:
                mark_not_available(
                    plan, sub_goal,
                    outcome.reason or f"navigation ended {outcome.nav_status}",
                )

            if outcome.status == "resolved" and self._may_expand(plan, sub_goal):
                before = len(plan.sub_goals)
                expand_plan(plan, self._narrow_to_named_entities(outcome.verdict, task),
                            pb_config)
                added = plan.sub_goals[before:]
                for sg in added:
                    sg.metadata["expansion_depth"] = (
                        sub_goal.metadata.get("expansion_depth", 0) + 1
                    )
                if added:
                    self._emit(
                        "plan_expanded",
                        parent_id=sub_goal.id,
                        added=[{"id": sg.id, "question": sg.question} for sg in added],
                        new_total=len(plan.sub_goals),
                    )

        self._emit("budget", **self._budget())
        self._synthesise(task, plan, validated_results, result)

        result.pages_used = self._pages_used
        result.llm_calls_used = self._llm.calls
        result.visited_urls = list(self._visited)
        result.gate_rejections = self._gate_rejections
        result.elapsed_s = time.monotonic() - started
        return result

    # -- one sub-goal ------------------------------------------------------
    def _run_sub_goal(self, plan: Any, sub_goal: Any, result: LoopResult) -> SubGoalOutcome:
        index = [sg.id for sg in plan.sub_goals].index(sub_goal.id) + 1
        self._emit(
            "sub_goal_started",
            sub_goal_id=sub_goal.id,
            question=sub_goal.question,
            index=index,
            total=len(plan.sub_goals),
            target_fields=list(sub_goal.target_fields or []),
        )

        # The remaining global page budget caps this navigation, so one sub-goal
        # cannot spend the plan's whole allowance.
        remaining_pages = max(1, self._max_pages - self._pages_used)
        navigator = Navigator(
            self._llm,
            fetcher=self._fetcher,
            step_logger=self._step_logger,
            language=self._language,
            max_pages=min(config.MAX_PAGES, remaining_pages),
        )

        collected = _Verdicts()
        try:
            nav = navigator.navigate(
                sub_goal.question,
                validate_fn=self._make_validate_fn(sub_goal, collected),
                exclude_urls=self._unreadable_urls(result),
            )
        except LLMError as exc:
            if result.error_reason is None:
                result.error_reason = str(exc)
            outcome = SubGoalOutcome(
                sub_goal_id=sub_goal.id, question=sub_goal.question,
                status="not_available", reason=f"model unavailable: {exc}",
            )
            self._emit("sub_goal_finished", **outcome.to_dict())
            return outcome

        self._pages_used += nav.pages_fetched
        result.trail.extend(nav.trail)
        for step in nav.trail:
            if step.fetch_ok and step.url not in self._visited:
                self._visited.append(step.url)
        if nav.status == "blocked" and result.blocked_reason is None:
            result.blocked_reason = nav.final_reasoning or "the site blocked the request"
        if nav.status == "error" and result.error_reason is None:
            result.error_reason = nav.final_reasoning or "the model could not be reached"

        outcome = self._outcome_from(sub_goal, nav, collected)
        self._fallback_extract(sub_goal, nav, collected, outcome)
        self._emit("sub_goal_finished", **outcome.to_dict(), **self._budget())
        return outcome

    # -- the extraction fallback ------------------------------------------
    @staticmethod
    def _has_usable_facts(outcome: SubGoalOutcome) -> bool:
        """Did this sub-goal end holding any field with a value in it?"""
        if outcome.status != "resolved":
            return False
        extracted = (outcome.verdict or {}).get("extracted") or {}
        return bool(
            any(
                (table.get("records") or [])
                for table in list(extracted.get("tables") or [])
                + list(extracted.get("pdf_tables") or [])
            )
            or extracted.get("fields")
        )

    @classmethod
    def _needs_extraction(cls, outcome: SubGoalOutcome) -> bool:
        """Fire whenever the sub-goal produced no usable facts. Full stop.

        This was a whitelist of terminal statuses, and it missed the one that
        matters most in practice. ``arrived`` -- the navigator reached a page
        and nothing validated it -- lands as ``not_available`` and was covered,
        but ``unreadable`` was not, and any status added later would not have
        been either. A whitelist of the ways to fail is a list that will be
        incomplete again.

        So the question is asked the other way round: are there facts? If not,
        read the page. That covers every route to an empty answer, including
        the contradiction of a green tick above "No verified facts were
        retrieved" -- a resolve on product names with no field extracted.

        Still never speculative: a resolve backed by real values returns False,
        because a second opinion there costs a call and can only agree.
        """
        return not cls._has_usable_facts(outcome)

    def _best_page_for(self, nav: NavigationResult, collected: "_Verdicts") -> tuple[str, str]:
        """The page most worth re-reading: where navigation stopped, else the last fetched.

        ``nav.page`` is consulted for its own text first rather than looking the
        URL up in ``self._page_text``. That cache is filled by ``validate_fn``,
        which the navigator skips for a page it could not fetch -- so relying on
        it made the fallback depend on a side effect that does not always
        happen, and when it did not the whole thing returned silently.
        """
        page = nav.page or {}
        url, text = page.get("url") or "", page.get("text") or ""
        if url and text:
            return url, text
        if url and self._page_text.get(url):
            return url, self._page_text[url]
        for step in reversed(nav.trail):
            if step.fetch_ok and self._page_text.get(step.url):
                return step.url, self._page_text[step.url]
        return "", ""

    def _fallback_extract(
        self, sub_goal: Any, nav: NavigationResult, collected: "_Verdicts",
        outcome: SubGoalOutcome,
    ) -> None:
        """Read the best page with the model, once, if the keyword path failed."""
        # Every exit from here says why, at INFO. A silent skip is how this
        # went unnoticed for a whole run: the log showed navigation ending and
        # the loop finishing 18ms later with nothing in between.
        if not self._fallback:
            return
        if not self._needs_extraction(outcome):
            logger.info(
                "extraction fallback not needed for %s: the sub-goal already holds facts",
                sub_goal.id,
            )
            return
        if nav.status == "blocked":
            logger.info("extraction fallback skipped for %s: the site blocked the run",
                        sub_goal.id)
            return
        if self._exhausted() is not None:
            logger.info("extraction fallback skipped for %s: budget spent", sub_goal.id)
            return

        # A run that never got past the seed has nothing worth re-reading: the
        # homepage answers no question, and reading it costs a call to find
        # that out.
        if nav.pages_fetched <= 1 and nav.status in ("no_candidates", "error", "blocked"):
            logger.info("extraction fallback skipped for %s: never left the seed", sub_goal.id)
            return

        url, text = self._best_page_for(nav, collected)
        if not text:
            logger.info(
                "extraction fallback skipped for %s: nothing to read (nav ended %s "
                "with %d page(s) and no readable text)",
                sub_goal.id, nav.status, nav.pages_fetched,
            )
            return

        logger.info(
            "extraction fallback firing for %s (nav ended %s): re-reading %s (%d chars)",
            sub_goal.id, nav.status, url, len(text),
        )

        extraction = extract_facts(sub_goal.question, text, self._llm)
        outcome.extraction = extraction.to_dict()
        self._emit("extraction", sub_goal_id=sub_goal.id, url=url, **extraction.to_dict())
        if not extraction.used:
            return

        # The facts become a verdict in the shape the rest of the pipeline
        # already reads. Their coverage logic is deliberately bypassed: it is
        # the thing that just failed, and re-asking it would only fail again.
        verdict = dict(outcome.verdict or {})
        extracted = dict(verdict.get("extracted") or {})
        extracted["tables"] = list(extracted.get("tables") or []) + [
            extraction.as_table(sub_goal.question)
        ]
        verdict.update(
            resolved=True,
            status="resolved",
            extracted=extracted,
            source_url=verdict.get("source_url") or url,
            reason=(
                f"read from the page text by the model: {extraction.reason} "
                f"(keyword extraction found none)"
            ),
        )

        decision = self._gate.judge(verdict, url, text)
        if not decision.accepted:
            self._gate_rejections += 1
            logger.info("acceptance gate withheld an extraction-fallback resolve: %s",
                        decision.reason)
            return

        outcome.verdict = verdict
        outcome.status = "resolved"
        outcome.reason = verdict["reason"]
        outcome.source_url = verdict["source_url"]
        outcome.gate = decision

    def _outcome_from(
        self, sub_goal: Any, nav: NavigationResult, collected: "_Verdicts"
    ) -> SubGoalOutcome:
        # On a resolve the verdict is the one for the page navigation stopped
        # on. Otherwise take the best verdict seen anywhere on the walk, which
        # is how a partial survives a run that ended ``exhausted``.
        page_url = (nav.page or {}).get("url")
        verdict = collected.by_url.get(page_url or "") or collected.best

        status = "unresolved"
        reason = nav.final_reasoning or ""
        if nav.status == "resolved" and verdict is not None and verdict.get("resolved"):
            status = "resolved"
            reason = verdict.get("reason", "")
        elif verdict is not None and verdict.get("status") in ("partial", "unreadable"):
            status = verdict["status"]
            reason = verdict.get("reason", "")
        elif nav.status in ("exhausted", "no_candidates", "arrived"):
            status = "not_available"

        gate = verdict.get("gate") if verdict else None
        return SubGoalOutcome(
            sub_goal_id=sub_goal.id,
            question=sub_goal.question,
            status=status,
            reason=reason,
            source_url=(verdict or {}).get("source_url") or page_url,
            verdict=verdict,
            nav_status=nav.status,
            hops_used=nav.hops_used,
            pages_fetched=nav.pages_fetched,
            gate=GateDecision(**gate) if isinstance(gate, dict) else None,
        )

    # -- plan bookkeeping --------------------------------------------------
    # Task types whose answer genuinely requires investigating several entities
    # separately. Everything else -- a lookup, a how-to, a single-entity fee
    # question -- is answered by one page and must not fan out.
    EXPANDING_TASK_TYPES = frozenset({"comparison", "recommendation", "multi_hop"})

    def _may_expand(self, plan: Any, sub_goal: Any) -> bool:
        """Should a resolved sub-goal spawn one per discovered entity?

        Almost always no. ``expand_plan`` reads the entity list off whatever
        page resolved and creates a sub-goal for each, without consulting the
        task -- and this loop used to call it on every resolve. Measured, that
        turned "What are the fees on the Classic credit card?" into four
        sub-goals: the answer, then one each for Gold, Platinum and Titanium,
        spending the model-call budget on cards nobody asked about and listing
        them under "Not found".

        The task type is the gate. It scores 14/14 on the labelled set in
        ``tests/test_loop.py::TestExpansionGate``; re-run it if this changes.
        """
        if len(plan.sub_goals) >= self._max_sub_goals:
            return False
        if sub_goal.metadata.get("expansion_depth", 0) >= self._max_expansion_depth:
            return False
        # Model planning and expansion do the same job by different means, and
        # running both means two mechanisms competing for one budget and
        # producing near-duplicate sub-goals -- the planner writing "the fees
        # of the Classic card" while expansion adds "Find fees for Classic
        # Credit Card" off the page it landed on.
        #
        # So: if the planner decomposed, it has already done the job. If it
        # returned a single sub-goal, expansion is the fallback that finds what
        # the planner could not know before anything was fetched -- which is
        # most of what expansion is for.
        if self._planned_sub_goals > 1:
            return False
        return plan.goal.task_type in self.EXPANDING_TASK_TYPES

    @staticmethod
    def _narrow_to_named_entities(verdict: dict[str, Any], task: str) -> dict[str, Any]:
        """Expand only to the entities the task actually named, if it named any.

        "Compare the Classic and Gold credit cards" discovers eight cards and
        would fan out to all of them; ``MAX_SUB_GOALS`` then truncates to an
        arbitrary three, which may not include either card asked about. When
        the task names entities that were discovered, those are the ones worth
        a sub-goal. When it names none -- an open comparison or a
        recommendation -- the full set is right and is left alone.
        """
        entities = ((verdict.get("extracted") or {}).get("entities")) or []
        if not entities:
            return verdict
        usable = [e for e in entities if isinstance(e, dict) and e.get("name")]
        singled_out = set(_entities_named_by(task, [str(e["name"]) for e in usable]))
        if not singled_out or len(singled_out) == len(usable):
            return verdict
        named = [e for e in usable if str(e["name"]) in singled_out]
        logger.info(
            "expansion narrowed to the entities the task named: %s (of %d discovered)",
            [e["name"] for e in named], len(entities),
        )
        return {**verdict, "extracted": {**verdict["extracted"], "entities": named}}

    def _mark_remaining_unavailable(self, plan: Any, reason: str, result: LoopResult) -> None:
        from person_b.api import mark_not_available
        from person_b.models import SubGoalStatus

        for sg in plan.sub_goals:
            if sg.status == SubGoalStatus.PENDING:
                mark_not_available(plan, sg, reason)
                outcome = SubGoalOutcome(
                    sub_goal_id=sg.id, question=sg.question,
                    status="not_available", reason=reason,
                )
                result.outcomes.append(outcome)
                self._emit("sub_goal_finished", **outcome.to_dict(), **self._budget())

    @staticmethod
    def _unreadable_urls(result: LoopResult) -> list[str]:
        """Don't send the navigator back to a document it could not read.

        The image-only PDF case: a scanned document yields no text however many
        times it is fetched, so re-selecting it just burns a hop.
        """
        return [o.source_url for o in result.outcomes
                if o.status == "unreadable" and o.source_url]

    # -- answer ------------------------------------------------------------
    def _synthesise(
        self, task: str, plan: Any, validated: list[dict[str, Any]], result: LoopResult
    ) -> None:
        from person_b.api import finalize, synthesize, validate_answer

        self._emit("synthesis_started", validated_results=len(validated),
                   visited_pages=len(self._visited))

        synthesis = synthesize(task, plan=plan, validated_results=validated)
        claims = synthesis.get("claims", [])

        # validate_answer is given exactly the pages this run fetched -- and
        # their *text*, not just their URLs. Person B's attribution accepts
        # either, but with bare URLs it can only check "was this page visited",
        # which every mechanically-built claim passes by construction. Handing
        # it the content is what turns the support rate from "cites a page we
        # fetched" into "its value is on the page it cites" (PATCH 17).
        visited_pages = [
            {"url": url, "text": self._page_text.get(url, "")} for url in self._visited
        ]
        verification = validate_answer(claims, visited_pages, strict=True)
        final = finalize(synthesis, verification)
        meta = final.get("metadata", {})

        result.answer = final.get("answer", "")
        result.source_urls = list(final.get("source_urls", []))
        result.not_found = list(final.get("not_found", []))
        result.support_rate = float(meta.get("support_rate", 1.0))
        result.claims_total = int(meta.get("claims_total", 0))
        result.claims_supported = int(meta.get("claims_supported", 0))

        result.template_answer = result.answer

        self._emit(
            "answer_verified",
            support_rate=result.support_rate,
            claims_total=result.claims_total,
            claims_supported=result.claims_supported,
            claims_unsupported=int(meta.get("claims_unsupported", 0)),
            claims_contradicted=int(meta.get("claims_contradicted", 0)),
            prose_removed=list(meta.get("prose_removed", [])),
        )

        self._compose_answer(task, verification, result)
        logger.info(
            "loop finished: task=%r sub_goals=%d resolved=%d support=%.2f "
            "pages=%d llm_calls=%d gate_rejections=%d answer=%s",
            task, len(result.outcomes), result.resolved_count, result.support_rate,
            self._pages_used, self._llm.calls, self._gate_rejections,
            result.answer_source,
        )

    def _compose_answer(self, task: str, verification: dict, result: LoopResult) -> None:
        """Replace the template wording with grounded prose, if that is possible.

        The template answer is already in ``result.answer`` before this runs and
        stays there unless composition produces something that survives
        grounding. Composition can improve the answer; it can never remove one.
        """
        if not self._compose:
            return

        # Only claims that passed verification are offered. Composing from the
        # unfiltered set would let a claim their attribution rejected reappear
        # in the prose, laundered through the model.
        verified = [
            chk.get("claim") or {}
            for chk in verification.get("passed", [])
            if isinstance(chk, dict)
        ]
        composition = (
            compose_tiered(task, verified, self._llm, language=self._language,
                           task_type=result.task_type)
            if self._tiers else
            compose_answer(task, verified, self._llm, language=self._language)
        )
        result.composition = composition.to_dict()

        if not composition.attempted:
            self._emit("answer_composed", **composition.to_dict())
            return

        if composition.used:
            result.answer = composition.text
            result.answer_source = "composed"
            result.answer_tiers = dict(composition.tiers)
        self._emit("answer_composed", **composition.to_dict())
