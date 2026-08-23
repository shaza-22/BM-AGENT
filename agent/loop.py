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
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from agent import config
from agent.acceptance import AcceptanceGate, GateDecision, gate_from_config
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
            "resolved_count": self.resolved_count,
            "elapsed_s": round(self.elapsed_s, 2),
        }


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

        self._pages_used = 0
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
        if self._llm.calls >= self._max_llm_calls:
            return f"model-call budget spent ({self._llm.calls}/{self._max_llm_calls})"
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

        plan = plan_task(task, pb_config)
        result = LoopResult(task=task, plan_id=plan.id, task_type=plan.goal.task_type)
        self._emit(
            "plan",
            plan_id=plan.id,
            task_type=plan.goal.task_type,
            target_fields=plan.goal.metadata.get("target_fields", []),
            sub_goals=[{"id": sg.id, "question": sg.question, "status": sg.status.value}
                       for sg in plan.sub_goals],
            gate_enabled=self._gate.enabled,
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
                expand_plan(plan, outcome.verdict, pb_config)
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
        self._emit("sub_goal_finished", **outcome.to_dict(), **self._budget())
        return outcome

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
    def _may_expand(self, plan: Any, sub_goal: Any) -> bool:
        if len(plan.sub_goals) >= self._max_sub_goals:
            return False
        return sub_goal.metadata.get("expansion_depth", 0) < self._max_expansion_depth

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

        # validate_answer is given exactly the pages this run fetched, which is
        # what makes the check meaningful: a claim citing anything else cannot
        # be attributed and does not survive into the answer.
        verification = validate_answer(claims, self._visited, strict=True)
        final = finalize(synthesis, verification)
        meta = final.get("metadata", {})

        result.answer = final.get("answer", "")
        result.source_urls = list(final.get("source_urls", []))
        result.not_found = list(final.get("not_found", []))
        result.support_rate = float(meta.get("support_rate", 1.0))
        result.claims_total = int(meta.get("claims_total", 0))
        result.claims_supported = int(meta.get("claims_supported", 0))

        self._emit(
            "answer_verified",
            support_rate=result.support_rate,
            claims_total=result.claims_total,
            claims_supported=result.claims_supported,
            claims_unsupported=int(meta.get("claims_unsupported", 0)),
            claims_contradicted=int(meta.get("claims_contradicted", 0)),
            prose_removed=list(meta.get("prose_removed", [])),
        )
        logger.info(
            "loop finished: task=%r sub_goals=%d resolved=%d support=%.2f "
            "pages=%d llm_calls=%d gate_rejections=%d",
            task, len(result.outcomes), result.resolved_count, result.support_rate,
            self._pages_used, self._llm.calls, self._gate_rejections,
        )
