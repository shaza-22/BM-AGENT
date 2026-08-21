"""
The navigation loop: get from the homepage to the page that answers a sub-goal.

What it does
    Starts at the seed URL, and repeatedly fetches a page, reads its links,
    checks whether the sub-goal is answered, and -- if not -- asks the link
    selector which link to follow next. Stops when the sub-goal resolves, when
    no candidate fits, when a cap is reached, or when the site blocks us.

Inputs
    ``navigate(sub_goal, validate_fn=None, exclude_urls=None)``.
    ``validate_fn(sub_goal, page) -> {"resolved": bool, "extracted": dict,
    "reason": str}`` receives the full ``PageDict`` (so it can use ``raw_html``
    for tables, ``text`` for prose and ``url`` for attribution). It defaults to
    a stub that never resolves, so the loop is exercisable before the real
    validator exists.

Outputs
    A :class:`NavigationResult` with the resolving page (if any), the full
    trail of hops including the reasoning behind each, and the resource counts.

Why it is needed
    This is the agentic core. Nothing here is scripted per task: the same loop
    serves a product lookup, a comparison and a multi-step research question,
    and the only thing that differs is what the model decides at each hop.

Search strategy -- frontier, not tree-walk
    Candidates accumulate across the whole run rather than only from the
    current page, so hop 4 may follow a link first seen on hop 1. This matters
    because the site repeats its entire navigation on every page (60 of 173
    links observed across saved pages appear on all of them): the link that
    resolves a sub-goal is often one that was already on screen two hops ago,
    and with a budget of five hops there is no room to re-walk a tree to reach
    it. Ranking keeps the default behaviour depth-first, and every step records
    ``from_url`` and ``discovered_at_hop`` so a lateral jump is legible in the
    log rather than mysterious.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

from agent import config
from agent.link_selector import (
    Candidate,
    Selection,
    SelectionContext,
    select_next_link,
)
from agent.llm import LLMClient, LLMError
from agent.messages import message
from agent.trail_log import StepLogger
from browsing.extract_links import alias_key, canonical_key, extract_links, normalize_url
from browsing import config as browsing_config
from browsing.fetcher import Fetcher, PageDict

logger = logging.getLogger(__name__)

NavigationStatus = Literal[
    "resolved", "arrived", "exhausted", "no_candidates", "blocked", "error"
]
ValidateFn = Callable[[str, PageDict], dict]


@dataclass
class TrailStep:
    """One page the agent actually visited, and why it went there."""

    hop: int
    url: str
    label: str | None
    source: str | None
    from_url: str | None
    discovered_at_hop: int | None
    reasoning: str
    confidence: float
    fetch_ok: bool
    fetch_status: int
    content_type: str
    validated: bool
    validate_reason: str
    candidates_offered: int
    candidates_available: int
    links_found: int
    pages_fetched: int
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NavigationResult:
    """The outcome of navigating one sub-goal.

    ``status`` values:

    ``resolved``
        ``validate_fn`` confirmed the page answers the sub-goal. This is the
        only status that makes that claim, and only ``validate_fn`` can produce
        it -- the link selector never can.
    ``arrived``
        Navigation stopped because the selector judged the current page to be
        the destination, but no validator confirmed it. ``page`` is the page
        reached, so a caller can still extract from it; ``extracted`` is None.
        With the stub validator this is the normal outcome of a successful
        walk. Once a real validator is installed it should become rare, and
        each occurrence is worth inspecting: it means the navigator believed it
        had arrived and the validator disagreed.
    ``no_candidates``
        Nothing on the pages seen leads toward the sub-goal.
    ``exhausted`` / ``blocked`` / ``error``
        A cap was reached, the site returned a WAF block page, or the model
        could not be reached.
    """

    status: NavigationStatus
    page: PageDict | None
    trail: list[TrailStep]
    pages_fetched: int
    hops_used: int
    extracted: dict | None
    sub_goal: str = ""
    language: str | None = None
    cap_hit: Literal["hops", "pages"] | None = None
    final_reasoning: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        """URLs actually fetched this run -- the evidence for any claim made."""
        return [step.url for step in self.trail if step.fetch_ok]


def always_unresolved(sub_goal: str, page: PageDict) -> dict:
    """Default validator: never resolves, so navigation runs to its cap.

    A placeholder for the real validation/extraction layer. It deliberately
    does not inspect the page -- guessing here would mask the fact that no
    validator is installed.
    """
    return {
        "resolved": False,
        "extracted": {},
        "reason": "no validator installed (stub); navigation will run to its cap",
    }


class Navigator:
    """Runs one sub-goal at a time against a single :class:`Fetcher`."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        fetcher: Fetcher | None = None,
        validate_fn: ValidateFn | None = None,
        step_logger: StepLogger | None = None,
        seed_url: str | None = None,
        language: str | None = None,
        max_hops: int = config.MAX_HOPS,
        max_pages: int = config.MAX_PAGES,
        candidate_limit: int = config.CANDIDATE_LIMIT,
    ) -> None:
        self._llm = llm
        # One Fetcher for the whole run: the per-run cache and the politeness
        # clock both live on it. The site is behind a WAF that bans on burst
        # traffic, so fetches must stay strictly sequential.
        self._fetcher = fetcher if fetcher is not None else Fetcher()
        self._validate_fn = validate_fn
        self._log = step_logger if step_logger is not None else StepLogger()
        # The run's language decides both where it starts and which links it
        # will consider. Under the permissive policy the fetcher and extractor
        # keep every language and the ranker prefers this one; under "strict"
        # the other language is dropped before the model ever sees it.
        self.language = language if language is not None else browsing_config.LANGUAGE
        self._fetch_language = (
            self.language
            if config.CROSS_LANGUAGE_POLICY == "strict"
            else browsing_config.LANGUAGE_ANY
        )
        self._seed_url = seed_url if seed_url is not None else config.seed_for(self.language)
        # Set unconditionally, including on a fetcher the caller supplied. The
        # navigator owns the run's language; a passed-in Fetcher supplies
        # transport -- session, rate limiter, cache -- not language policy.
        # Guarding this with `if fetcher is None` meant every real caller
        # (the API runner and the CLI both pass a fetcher) ran with the
        # process default, so an Arabic run rejected its own Arabic seed.
        self._fetcher.language = self._fetch_language
        self._max_hops = max_hops
        self._max_pages = max_pages
        self._candidate_limit = candidate_limit

    @property
    def step_logger(self) -> StepLogger:
        return self._log

    # -- public ------------------------------------------------------------
    def navigate(
        self,
        sub_goal: str,
        validate_fn: ValidateFn | None = None,
        exclude_urls: Iterable[str] | None = None,
        familiar_keys: frozenset[str] = frozenset(),
    ) -> NavigationResult:
        """Walk from the seed to a page that answers ``sub_goal``.

        Only ``validate_fn`` can produce ``status="resolved"``. The selector's
        strongest claim is ``status="arrived"`` -- it sees link labels, never
        page content, so it can report that there is nowhere better to go but
        never that the sub-goal is answered.

        ``familiar_keys`` are pages seen in earlier turns of a conversation.
        They only nudge the ranking, and only when
        :data:`agent.config.FOLLOW_UP_PATH_BONUS` is non-zero (it ships at
        zero). Nothing here can be fetched without first being discovered from
        the seed this run.
        """
        validate = validate_fn or self._validate_fn or always_unresolved

        visited: set[str] = set()
        alias_visited: set[str] = set()
        frontier: dict[str, Candidate] = {}
        trail: list[TrailStep] = []

        # Keys the caller has already consumed elsewhere (an earlier sub-goal,
        # or a retry that should take a different route).
        for url in exclude_urls or ():
            self._mark_visited(url, visited, alias_visited)

        self._log.emit(
            "navigation_started",
            sub_goal=sub_goal,
            language=self.language,
            seed=self._seed_url,
            max_hops=self._max_hops,
            max_pages=self._max_pages,
        )

        run_started = time.perf_counter()
        nav_link_extract_s = 0.0
        # A run that cannot fetch its own starting page is a configuration
        # error, not a dead end, and it should say so rather than reporting
        # "no candidates" three lines later.
        if normalize_url(self._seed_url, self._seed_url, language=self._fetch_language) is None:
            logger.error(
                "seed %s is rejected by this run's own filters (language=%r, fetch=%r)",
                self._seed_url, self.language, self._fetch_language,
            )

        pages_fetched = 0
        hops_used = 0
        current_page: PageDict | None = None
        pending: Selection | None = None
        next_url: str | None = self._seed_url

        while next_url is not None:
            started = time.perf_counter()
            page = self._fetcher.fetch(next_url)
            pages_fetched += 1
            self._mark_visited(next_url, visited, alias_visited)
            self._mark_visited(page["url"], visited, alias_visited)

            link_started = time.perf_counter()
            links = (
                extract_links(page["raw_html"], page["url"], language=self._fetch_language)
                if page["ok"] and page["raw_html"]
                else []
            )
            # The fetcher already extracted these once for the escalation
            # check; re-extracting costs ~55ms on a 300KB page. Counted here
            # rather than hidden, so the duplication is visible if it ever
            # matters.
            nav_link_extract_s += time.perf_counter() - link_started

            if page["ok"] and self._looks_blocked(page, len(links)):
                # Continuing would hammer a WAF that has already flagged us,
                # which is how a run becomes a ban. Stop the whole navigation.
                step = self._make_step(
                    trail, self.language, page, pending, links, False,
                    "site returned a WAF block page", pages_fetched, started,
                )
                trail.append(step)
                self._log.emit("step", **step.to_dict())
                logger.warning("WAF block page detected at %s -- aborting run", page["url"])
                return self._finish(
                    "blocked", None, trail, pages_fetched, hops_used, None, sub_goal, run_started=run_started, nav_link_extract_s=nav_link_extract_s,
                    final_reasoning=message("blocked", self.language),
                )

            verdict = self._validate(validate, sub_goal, page)
            resolved = bool(verdict.get("resolved")) and page["ok"]

            step = self._make_step(
                trail, self.language, page, pending, links, resolved,
                str(verdict.get("reason") or ""), pages_fetched, started,
            )
            trail.append(step)
            self._log.emit("step", **step.to_dict())

            if resolved:
                return self._finish(
                    "resolved", page, trail, pages_fetched, hops_used,
                    verdict.get("extracted") or {}, sub_goal, run_started=run_started, nav_link_extract_s=nav_link_extract_s,
                    final_reasoning=str(verdict.get("reason") or "sub-goal resolved"),
                )

            if page["ok"]:
                current_page = page
                self._grow_frontier(frontier, links, page["url"], len(trail) - 1, visited)
            else:
                # A dead end is not fatal: the frontier still holds every link
                # from earlier pages, so the next selection can go elsewhere.
                logger.info("dead end at %s (%s)", page["url"], page["error"])

            if hops_used >= self._max_hops:
                return self._finish(
                    "exhausted", None, trail, pages_fetched, hops_used, None, sub_goal, run_started=run_started, nav_link_extract_s=nav_link_extract_s,
                    cap_hit="hops",
                    final_reasoning=message("cap_hops", self.language, cap=self._max_hops),
                )
            if pages_fetched >= self._max_pages:
                return self._finish(
                    "exhausted", None, trail, pages_fetched, hops_used, None, sub_goal, run_started=run_started, nav_link_extract_s=nav_link_extract_s,
                    cap_hit="pages",
                    final_reasoning=message("cap_pages", self.language, cap=self._max_pages),
                )

            context = SelectionContext(
                hop=hops_used,
                pages_fetched=pages_fetched,
                max_hops=self._max_hops,
                max_pages=self._max_pages,
                current_url=current_page["url"] if current_page else None,
                trail=[f"{s.label or 'start'} -> {s.url}" for s in trail],
                alias_visited=frozenset(alias_visited),
                familiar_keys=familiar_keys,
                language=self.language,
                limit=self._candidate_limit,
            )

            try:
                selection = select_next_link(
                    sub_goal, list(frontier.values()), visited, context, llm=self._llm
                )
            except LLMError as exc:
                logger.error("link selection failed: %s", exc)
                return self._finish(
                    "error", None, trail, pages_fetched, hops_used, None, sub_goal, run_started=run_started, nav_link_extract_s=nav_link_extract_s,
                    final_reasoning=message("llm_unreachable", self.language, error=exc),
                )

            self._log.emit(
                "selection",
                hop=hops_used + 1,
                sub_goal=sub_goal,
                chosen_url=selection.url,
                chosen_label=selection.label,
                candidate_index=selection.candidate_index,
                outcome=selection.outcome,
                elapsed_ms=selection.elapsed_ms,
                reasoning=selection.reasoning,
                confidence=selection.confidence,
                candidates_offered=selection.offered,
                candidates_available=selection.available,
                parse_error=selection.parse_error,
            )

            # Every offered-but-unchosen link is demoted, not removed: it was
            # not rejected, only out-ranked, and it is the natural next try.
            for index, candidate in enumerate(selection.ranked):
                if index != selection.candidate_index:
                    candidate.times_offered += 1

            if selection.url is None:
                # "arrived" is the selector reporting that the route ends here,
                # not that the sub-goal is answered -- so the page comes back
                # for extraction, but extracted stays None and the status is
                # never "resolved". It needs a page that actually loaded; if the
                # last fetch failed there is nothing to have arrived at.
                if selection.outcome == "arrived" and current_page is not None:
                    return self._finish(
                        "arrived", current_page, trail, pages_fetched, hops_used, None, sub_goal, run_started=run_started, nav_link_extract_s=nav_link_extract_s,
                        final_reasoning=selection.reasoning,
                    )
                return self._finish(
                    "no_candidates", None, trail, pages_fetched, hops_used, None, sub_goal, run_started=run_started, nav_link_extract_s=nav_link_extract_s,
                    final_reasoning=selection.reasoning,
                )

            if self._alias_already_visited(selection.url, alias_visited):
                logger.warning(
                    "following %s although its alternate URL scheme was already visited",
                    selection.url,
                )

            pending = selection
            hops_used += 1
            next_url = selection.url

        # Unreachable: the loop only exits through a return above.
        return self._finish(
            "error", None, trail, pages_fetched, hops_used, None, sub_goal,
            run_started=run_started, nav_link_extract_s=nav_link_extract_s,
            final_reasoning="Navigation ended without a decision.",
        )

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _mark_visited(url: str, visited: set[str], alias_visited: set[str]) -> None:
        normalized = normalize_url(url, url)
        if not normalized:
            return
        visited.add(canonical_key(normalized))
        alias = alias_key(normalized)
        if alias:
            alias_visited.add(alias)

    @staticmethod
    def _alias_already_visited(url: str, alias_visited: set[str]) -> bool:
        alias = alias_key(url)
        return bool(alias and alias in alias_visited)

    @staticmethod
    def _looks_blocked(page: PageDict, link_count: int) -> bool:
        """Spot a WAF refusal served as HTTP 200.

        Requires both a marker and a near-empty link list: a real page carries
        70+ links, so the pair together will not fire on an article that merely
        uses the words.
        """
        if link_count > config.WAF_MAX_LINKS:
            return False
        text = (page.get("text") or "").lower()
        return any(marker in text for marker in config.WAF_BLOCK_MARKERS)

    @staticmethod
    def _validate(validate: ValidateFn, sub_goal: str, page: PageDict) -> dict:
        if not page["ok"]:
            return {"resolved": False, "extracted": {}, "reason": f"fetch failed: {page['error']}"}
        try:
            verdict = validate(sub_goal, page)
        except Exception as exc:
            # A validator bug must cost one page, not the whole task.
            logger.exception("validate_fn raised on %s", page["url"])
            return {"resolved": False, "extracted": {}, "reason": f"validator raised: {exc}"}
        if not isinstance(verdict, dict):
            return {
                "resolved": False,
                "extracted": {},
                "reason": f"validator returned {type(verdict).__name__}, expected dict",
            }
        return verdict

    @staticmethod
    def _grow_frontier(
        frontier: dict[str, Candidate],
        links: Sequence[dict],
        page_url: str,
        hop: int,
        visited: set[str],
    ) -> None:
        for link in links:
            key = link["key"]
            if key in visited or key in frontier:
                # Keeping the first sighting preserves the staleness decay --
                # a link re-offered by every page's nav must not look fresh.
                continue
            frontier[key] = Candidate(
                link=link, discovered_on=page_url, discovered_at_hop=hop  # type: ignore[arg-type]
            )

    @staticmethod
    def _make_step(
        trail: list[TrailStep],
        language: str | None,
        page: PageDict,
        pending: Selection | None,
        links: Sequence[dict],
        resolved: bool,
        validate_reason: str,
        pages_fetched: int,
        started: float,
    ) -> TrailStep:
        chosen: Candidate | None = None
        if pending and pending.candidate_index is not None and pending.ranked:
            if 0 <= pending.candidate_index < len(pending.ranked):
                chosen = pending.ranked[pending.candidate_index]
        return TrailStep(
            hop=len(trail),
            url=page["url"],
            label=pending.label if pending else None,
            source=chosen.source if chosen else None,
            from_url=chosen.discovered_on if chosen else None,
            discovered_at_hop=chosen.discovered_at_hop if chosen else None,
            reasoning=pending.reasoning if pending else message("seed_step", language),
            confidence=pending.confidence if pending else 1.0,
            fetch_ok=page["ok"],
            fetch_status=page["status"],
            content_type=page["content_type"],
            validated=resolved,
            validate_reason=validate_reason,
            candidates_offered=pending.offered if pending else 0,
            candidates_available=pending.available if pending else 0,
            links_found=len(links),
            pages_fetched=pages_fetched,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def _run_timing(self, run_started: float, nav_link_extract_s: float = 0.0) -> dict[str, float]:
        """Where the run's wall-clock went.

        Reported because "the run took three minutes" is not actionable on its
        own: politeness pacing, retry backoff, a slow site and a slow model all
        look identical from outside, and only one of them is ours to change.
        """
        fetcher = self._fetcher.stats
        return {
            "total_s": round(time.perf_counter() - run_started, 2),
            "page_request_s": fetcher.get("request_s", 0.0),
            "page_rate_limit_wait_s": fetcher.get("rate_limit_wait_s", 0.0),
            "page_retry_wait_s": fetcher.get("retry_wait_s", 0.0),
            "robots_s": fetcher.get("robots_s", 0.0),
            "pdf_text_s": fetcher.get("pdf_text_s", 0.0),
            "html_parse_s": fetcher.get("html_parse_s", 0.0),
            "link_extract_s": round(
                fetcher.get("link_extract_s", 0.0) + nav_link_extract_s, 3
            ),
            "llm_s": round(getattr(self._llm, "api_seconds", 0.0) or 0.0, 3),
            "llm_retry_wait_s": round(getattr(self._llm, "retry_wait_s", 0.0) or 0.0, 3),
            "llm_retries": getattr(self._llm, "retries", 0) or 0,
        }

    def _finish(
        self,
        status: NavigationStatus,
        page: PageDict | None,
        trail: list[TrailStep],
        pages_fetched: int,
        hops_used: int,
        extracted: dict | None,
        sub_goal: str,
        *,
        cap_hit: Literal["hops", "pages"] | None = None,
        final_reasoning: str = "",
        run_started: float | None = None,
        nav_link_extract_s: float = 0.0,
    ) -> NavigationResult:
        stats = {
            "fetcher": self._fetcher.stats,
            "llm_calls": getattr(self._llm, "calls", None),
            "timing": self._run_timing(
                run_started if run_started is not None else time.perf_counter(),
                nav_link_extract_s,
            ),
        }
        self._log.emit(
            "navigation_finished",
            sub_goal=sub_goal,
            status=status,
            language=self.language,
            cap_hit=cap_hit,
            pages_fetched=pages_fetched,
            hops_used=hops_used,
            final_reasoning=final_reasoning,
            stats=stats,
        )
        logger.info(
            "navigation %s for %r after %d hops / %d pages%s",
            status, sub_goal, hops_used, pages_fetched,
            f" (cap: {cap_hit})" if cap_hit else "",
        )
        return NavigationResult(
            status=status,
            page=page,
            trail=trail,
            pages_fetched=pages_fetched,
            hops_used=hops_used,
            extracted=extracted,
            sub_goal=sub_goal,
            language=self.language,
            cap_hit=cap_hit,
            final_reasoning=final_reasoning,
            stats=stats,
        )


def navigate(
    sub_goal: str,
    *,
    llm: LLMClient,
    validate_fn: ValidateFn | None = None,
    exclude_urls: Iterable[str] | None = None,
    **kwargs: Any,
) -> NavigationResult:
    """Navigate one sub-goal with a fresh :class:`Navigator`.

    Convenience for scripts and one-off calls. A task with several sub-goals
    should build one ``Navigator`` (and so one ``Fetcher``) and reuse it, so
    the per-run cache and politeness clock apply across the whole task.
    """
    return Navigator(llm, validate_fn=validate_fn, **kwargs).navigate(
        sub_goal, exclude_urls=exclude_urls
    )
