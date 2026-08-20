"""Tests for candidate ranking, prompt construction and reply parsing.

The most important test here is
``TestTopicAgnosticism::test_ranking_is_independent_of_the_sub_goal``: it is
what makes "no lexical matching" a checkable property rather than a promise.
"""

from __future__ import annotations

import json

import pytest

from agent import config
from agent.link_selector import (
    Candidate,
    SelectionContext,
    build_prompt,
    parse_selection,
    rank_candidates,
    score_candidate,
    select_next_link,
)
from agent.llm import FakeLLMClient
from browsing.extract_links import canonical_key

HOME = "https://www.banquemisr.com/"
PAGE_A = "https://www.banquemisr.com/Home/Pages/A"


def make_candidate(path, label="Label", source="body", hop=0, on=HOME, offered=0, is_pdf=False):
    url = path if path.startswith("http") else f"https://www.banquemisr.com{path}"
    link = {
        "label": label,
        "url": url,
        "key": canonical_key(url),
        "source": source,
        "is_pdf": is_pdf,
    }
    return Candidate(link=link, discovered_on=on, discovered_at_hop=hop, times_offered=offered)


def reply(choice, reasoning="because", confidence=0.8):
    return json.dumps({"choice": choice, "reasoning": reasoning, "confidence": confidence})


class TestScoring:
    def test_links_on_the_current_page_outrank_carried_over_ones(self):
        context = SelectionContext(hop=2, current_url=PAGE_A)
        here = make_candidate("/here", on=PAGE_A, hop=2)
        there = make_candidate("/there", on=HOME, hop=0)
        assert score_candidate(here, context) > score_candidate(there, context)

    def test_older_discoveries_decay_but_never_vanish(self):
        context = SelectionContext(hop=4, current_url=PAGE_A)
        recent = make_candidate("/recent", on=HOME, hop=3)
        old = make_candidate("/old", on=HOME, hop=0)
        assert score_candidate(recent, context) > score_candidate(old, context)

    def test_body_outranks_nav_after_the_first_hop(self):
        context = SelectionContext(hop=1, current_url=PAGE_A)
        body = make_candidate("/b", source="body", on=PAGE_A, hop=1)
        nav = make_candidate("/n", source="nav", on=PAGE_A, hop=1)
        assert score_candidate(body, context) > score_candidate(nav, context)

    def test_nav_is_not_penalised_on_the_first_hop(self):
        first = SelectionContext(hop=0, current_url=HOME)
        later = SelectionContext(hop=1, current_url=HOME)
        nav = make_candidate("/n", source="nav", on=HOME, hop=0)
        assert score_candidate(nav, first) > score_candidate(nav, later)

    def test_footer_is_kept_above_nav(self):
        # The fees hub is reachable only from the footer.
        context = SelectionContext(hop=2, current_url=PAGE_A)
        footer = make_candidate("/f", source="footer", on=PAGE_A, hop=2)
        nav = make_candidate("/n", source="nav", on=PAGE_A, hop=2)
        assert score_candidate(footer, context) > score_candidate(nav, context)

    def test_repeatedly_offered_links_decay_rather_than_disappear(self):
        context = SelectionContext(hop=3, current_url=PAGE_A)
        fresh = make_candidate("/fresh", on=HOME, hop=0, offered=0)
        stale = make_candidate("/stale", on=HOME, hop=0, offered=3)
        assert score_candidate(fresh, context) > score_candidate(stale, context)
        # Still rankable: a passed-over link is the natural next try after a
        # dead end, so it must never be filtered out entirely.
        assert stale in rank_candidates([fresh, stale], set(), context)

    def test_alias_of_a_visited_page_is_deprioritised_not_dropped(self):
        context = SelectionContext(
            hop=1, current_url=PAGE_A, alias_visited=frozenset({"banquemisr.com/about us/history"})
        )
        alias = make_candidate("/en/ABOUT-US/History", on=PAGE_A, hop=1)
        other = make_candidate("/en/ABOUT-US/Awards", on=PAGE_A, hop=1)
        assert score_candidate(alias, context) < score_candidate(other, context)
        assert alias in rank_candidates([alias, other], set(), context)


class TestRanking:
    def test_visited_links_are_dropped(self):
        context = SelectionContext(current_url=HOME)
        seen = make_candidate("/seen")
        fresh = make_candidate("/fresh")
        ranked = rank_candidates([seen, fresh], {seen.key}, context)
        assert [c.key for c in ranked] == [fresh.key]

    def test_list_is_capped(self):
        context = SelectionContext(current_url=HOME, limit=5)
        candidates = [make_candidate(f"/p{i}") for i in range(40)]
        assert len(rank_candidates(candidates, set(), context)) == 5

    def test_ties_keep_discovery_order(self):
        context = SelectionContext(current_url=HOME)
        candidates = [make_candidate(f"/p{i}", label=f"P{i}") for i in range(5)]
        ranked = rank_candidates(candidates, set(), context)
        assert [c.label for c in ranked] == ["P0", "P1", "P2", "P3", "P4"]

    def test_empty_input(self):
        assert rank_candidates([], set(), SelectionContext()) == []


class TestRegionQuotas:
    def test_a_body_heavy_page_still_offers_footer_links(self):
        # The real failure this guards: the homepage has 40 body links against
        # 11 footer ones, so a straight top-N cut offered zero footer links and
        # the footer-only fees hub could not be chosen on the first hop.
        context = SelectionContext(hop=0, current_url=HOME, limit=10)
        candidates = [make_candidate(f"/b{i}", source="body") for i in range(40)]
        candidates += [make_candidate(f"/f{i}", source="footer") for i in range(11)]
        ranked = rank_candidates(candidates, set(), context)

        assert len(ranked) == 10
        assert sum(1 for c in ranked if c.source == "footer") > 0

    def test_every_region_present_on_the_page_is_represented(self):
        context = SelectionContext(hop=1, current_url=HOME, limit=12)
        candidates = [make_candidate(f"/b{i}", source="body") for i in range(30)]
        candidates += [make_candidate(f"/n{i}", source="nav") for i in range(48)]
        candidates += [make_candidate(f"/f{i}", source="footer") for i in range(13)]
        sources = {c.source for c in rank_candidates(candidates, set(), context)}
        assert sources == {"body", "nav", "footer"}

    def test_quota_does_not_apply_when_everything_fits(self):
        context = SelectionContext(current_url=HOME, limit=50)
        candidates = [make_candidate(f"/b{i}", source="body") for i in range(5)]
        assert len(rank_candidates(candidates, set(), context)) == 5

    def test_result_stays_in_score_order(self):
        # Reserved links sit at their natural rank, not promoted to the top.
        context = SelectionContext(hop=1, current_url=HOME, limit=6)
        candidates = [make_candidate(f"/b{i}", source="body", on=HOME, hop=1) for i in range(10)]
        candidates += [make_candidate("/f0", source="footer", on=HOME, hop=1)]
        ranked = rank_candidates(candidates, set(), context)
        scores = [score_candidate(c, context) for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_a_region_absent_from_the_page_is_simply_skipped(self):
        context = SelectionContext(current_url=HOME, limit=3)
        candidates = [make_candidate(f"/b{i}", source="body") for i in range(10)]
        ranked = rank_candidates(candidates, set(), context)
        assert len(ranked) == 3
        assert all(c.source == "body" for c in ranked)


class TestTopicAgnosticism:
    def test_ranking_is_independent_of_the_sub_goal(self):
        # Ranking must never read the sub-goal: a lexical filter would quietly
        # become the agent, and would fail on any sub-goal worded differently
        # from the site's own vocabulary. score_candidate takes no sub-goal at
        # all, and this pins that down through the public entry point.
        context = SelectionContext(hop=1, current_url=PAGE_A)
        candidates = [
            make_candidate("/Home/Pages/Fees", label="Fees and Commissions", source="footer"),
            make_candidate("/Home/Pages/Cards", label="Credit Cards", source="body", on=PAGE_A, hop=1),
            make_candidate("/Home/Pages/Branches", label="Branches", source="nav"),
        ]
        baseline = [c.key for c in rank_candidates(candidates, set(), context)]

        for sub_goal in [
            "compare credit card fees",
            "what documents do I need to open an account",
            "recommend a card for frequent travel",
            "",
        ]:
            llm = FakeLLMClient([reply(0)])
            selection = select_next_link(sub_goal, candidates, set(), context, llm=llm)
            assert [c.key for c in selection.ranked] == baseline


class TestPrompt:
    def test_candidates_are_numbered_with_label_region_and_path(self):
        context = SelectionContext(hop=0, current_url=HOME)
        prompt = build_prompt("find the fees", [make_candidate("/Home/Pages/Fees", "Fees", "footer")], context)
        assert "0 | Fees | footer | /Home/Pages/Fees" in prompt
        assert "Sub-goal: find the fees" in prompt

    def test_host_is_not_repeated_per_line(self):
        context = SelectionContext(current_url=HOME)
        candidates = [make_candidate(f"/p{i}") for i in range(10)]
        prompt = build_prompt("goal", candidates, context)
        assert prompt.count("https://www.banquemisr.com") == 0

    def test_pdf_candidates_are_marked(self):
        context = SelectionContext(current_url=HOME)
        prompt = build_prompt("goal", [make_candidate("/t.pdf", "Tariff", is_pdf=True)], context)
        assert "[PDF]" in prompt

    def test_carried_over_candidates_say_where_they_came_from(self):
        context = SelectionContext(hop=3, current_url=PAGE_A)
        prompt = build_prompt("goal", [make_candidate("/old", on=HOME, hop=0)], context)
        assert "seen at hop 1" in prompt

    def test_budget_and_trail_are_included(self):
        context = SelectionContext(hop=1, pages_fetched=2, current_url=PAGE_A, trail=["start -> /"])
        prompt = build_prompt("goal", [make_candidate("/x")], context)
        assert "Hop 2 of 5" in prompt
        assert "2 of 15 pages used" in prompt
        assert "start -> /" in prompt


class TestParsing:
    def setup_method(self):
        self.ranked = [make_candidate("/a", "Alpha"), make_candidate("/b", "Beta")]

    def test_plain_json(self):
        selection = parse_selection(reply(1), self.ranked, available=9)
        assert selection.url.endswith("/b")
        assert selection.label == "Beta"
        assert selection.candidate_index == 1
        assert selection.available == 9
        assert selection.offered == 2

    def test_markdown_fences_are_stripped(self):
        selection = parse_selection(f"```json\n{reply(0)}\n```", self.ranked, 2)
        assert selection.label == "Alpha"

    def test_json_wrapped_in_prose_is_recovered(self):
        selection = parse_selection(f"Here is my answer:\n{reply(0)}\nHope that helps.", self.ranked, 2)
        assert selection.label == "Alpha"

    def test_minus_one_is_a_first_class_no_candidate(self):
        selection = parse_selection(reply(-1, "the site does not cover this"), self.ranked, 2)
        assert selection.url is None
        assert selection.candidate_index == -1
        assert selection.parse_error is None       # not an error, a decision
        assert "does not cover" in selection.reasoning

    def test_out_of_range_index_becomes_no_candidate(self):
        # An index cannot name a page that was never offered; an out-of-range
        # one is detectable, where a hallucinated URL would just be fetched.
        selection = parse_selection(reply(99), self.ranked, 2)
        assert selection.url is None
        assert "out of range" in selection.parse_error

    def test_malformed_json_becomes_no_candidate_with_the_error(self):
        selection = parse_selection("I think you should click Alpha!", self.ranked, 2)
        assert selection.url is None
        assert selection.parse_error
        assert "parse" in selection.reasoning.lower()
        assert selection.confidence == 0.0

    def test_numeric_string_choice_is_accepted(self):
        assert parse_selection('{"choice": "1", "reasoning": "x", "confidence": 0.5}',
                               self.ranked, 2).label == "Beta"

    def test_boolean_choice_is_rejected(self):
        # bool is an int subclass, so True would otherwise index candidate 1.
        selection = parse_selection('{"choice": true, "reasoning": "x", "confidence": 1}',
                                    self.ranked, 2)
        assert selection.url is None
        assert selection.parse_error

    def test_confidence_is_clamped_and_defaulted(self):
        assert parse_selection(reply(0, confidence=5), self.ranked, 2).confidence == 1.0
        assert parse_selection(reply(0, confidence=-2), self.ranked, 2).confidence == 0.0
        assert parse_selection('{"choice": 0, "reasoning": "x"}', self.ranked, 2).confidence == 0.5

    @pytest.mark.parametrize(
        "raw",
        [
            '{"choice": 0, "reasoning": "", "confidence": 0.9}',
            '{"choice": 0, "reasoning": "   ", "confidence": 0.9}',
            '{"choice": -1, "reasoning": "", "confidence": 0.9}',
            "not json at all",
            '{"choice": 42, "reasoning": "", "confidence": 0.9}',
        ],
    )
    def test_reasoning_is_never_empty(self, raw):
        # The reasoning field is a project deliverable -- it is shown to a human
        # explaining the agent's choice, so no path may leave it blank.
        assert parse_selection(raw, self.ranked, 2).reasoning.strip()


class TestSelectNextLink:
    def test_end_to_end_choice(self):
        context = SelectionContext(hop=0, current_url=HOME)
        candidates = [make_candidate("/a", "Alpha"), make_candidate("/b", "Beta")]
        llm = FakeLLMClient([reply(1, "Beta looks closer to the sub-goal")])
        selection = select_next_link("goal", candidates, set(), context, llm=llm)
        assert selection.label == "Beta"
        assert "Beta looks closer" in selection.reasoning
        assert llm.schemas[0]["properties"]["choice"]["type"] == "integer"

    def test_no_unvisited_candidates(self):
        candidate = make_candidate("/a")
        selection = select_next_link(
            "goal", [candidate], {candidate.key}, SelectionContext(), llm=FakeLLMClient([])
        )
        assert selection.url is None
        assert selection.offered == 0
        assert selection.reasoning

    def test_offered_and_available_are_reported(self):
        context = SelectionContext(current_url=HOME, limit=3)
        candidates = [make_candidate(f"/p{i}") for i in range(10)]
        selection = select_next_link("goal", candidates, set(), context, llm=FakeLLMClient([reply(0)]))
        assert selection.offered == 3
        assert selection.available == 10

    def test_low_confidence_is_discarded_when_a_minimum_is_configured(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.7)
        context = SelectionContext(current_url=HOME)
        llm = FakeLLMClient([reply(0, "not sure at all", confidence=0.2)])
        selection = select_next_link("goal", [make_candidate("/a")], set(), context, llm=llm)
        assert selection.url is None
        assert "below the configured minimum" in selection.reasoning

    def test_confidence_is_not_gated_by_default(self):
        context = SelectionContext(current_url=HOME)
        llm = FakeLLMClient([reply(0, "a guess", confidence=0.05)])
        selection = select_next_link("goal", [make_candidate("/a")], set(), context, llm=llm)
        assert selection.url is not None
