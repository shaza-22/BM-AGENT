"""Tests for conversation memory and follow-up resolution.

Offline: the resolver runs against FakeLLMClient, never a real model.
"""

from __future__ import annotations

import json
import logging

from agent import config
from agent.llm import FakeLLMClient, LLMError
from agent.navigator import NavigationResult, TrailStep
import pytest

from agent.session import (
    Resolution,
    Session,
    SessionStore,
    TurnSummary,
    build_resolver_prompt,
    parse_resolution,
    resolve_task,
)

HOME = "https://www.banquemisr.com/"
CARDS = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards"


def reply(sub_goal, used_context=True, reasoning="because"):
    return json.dumps(
        {"sub_goal": sub_goal, "used_context": used_context, "reasoning": reasoning}
    )


def step(hop, url, label, reasoning="why", source="body"):
    return TrailStep(
        hop=hop, url=url, label=label, source=source, from_url=None, discovered_at_hop=None,
        reasoning=reasoning, confidence=0.8, fetch_ok=True, fetch_status=200,
        content_type="html", validated=False, validate_reason="", candidates_offered=50,
        candidates_available=90, links_found=90, pages_fetched=hop + 1, elapsed_ms=100,
    )


def result(status="arrived", page_text="x" * 200_000):
    page = {
        "url": CARDS, "requested_url": CARDS, "status": 200, "ok": True,
        "content_type": "html", "raw_html": "<html>" + "y" * 300_000 + "</html>",
        "text": page_text, "fetched_at": "now", "render_mode": "requests", "error": None,
    }
    return NavigationResult(
        status=status, page=page,
        trail=[step(0, HOME, None), step(1, CARDS, "Cards")],
        pages_fetched=2, hops_used=1, extracted=None, sub_goal="find the cards",
        final_reasoning="reached the cards page",
        stats={"timing": {"total_s": 3.0}},
    )


class TestTurnSummary:
    def test_a_turn_keeps_no_page_content(self):
        # The reason this is a summary and not a NavigationResult: page text
        # reaches 140k characters and raw HTML 300KB. Holding those per turn
        # leaks memory and dumps hundreds of KB into anything that serialises.
        session = Session()
        session.add_turn("find the cards", Resolution("find the cards", False, "first"), result())
        turn = session.turns[0]

        assert isinstance(turn.summary, TurnSummary)
        blob = json.dumps(turn.summary, default=lambda o: o.__dict__)
        assert len(blob) < 2000
        assert "yyyy" not in blob and "xxxx" not in blob

    def test_a_turn_keeps_what_a_follow_up_needs(self):
        session = Session()
        session.add_turn("find the cards", Resolution("find the cards", False, "first"), result())
        summary = session.turns[0].summary

        assert summary.status == "arrived"
        assert summary.page_url == CARDS
        assert summary.sources == (HOME, CARDS)
        assert [hop.label for hop in summary.hops] == [None, "Cards"]

    def test_turns_are_capped(self):
        session = Session(max_turns=3)
        for index in range(6):
            session.add_turn(f"task {index}", Resolution(f"task {index}", False, "x"), result())
        assert len(session.turns) == 3
        assert session.turns[0].task == "task 3"      # oldest dropped


class TestResolution:
    def test_the_first_turn_passes_through_without_a_model_call(self):
        llm = FakeLLMClient([])
        resolution = resolve_task("find the credit cards", Session(), llm)
        assert resolution.sub_goal == "find the credit cards"
        assert resolution.used_context is False
        assert llm.calls == 0                          # no wasted request

    def test_a_follow_up_is_rewritten_using_prior_context(self):
        session = Session()
        session.add_turn("list the credit cards", Resolution("list the credit cards", False, "x"),
                         result())
        llm = FakeLLMClient([reply("find the fees for the classic credit card")])
        resolution = resolve_task("and the fees for that?", session, llm)

        assert resolution.sub_goal == "find the fees for the classic credit card"
        assert resolution.used_context is True
        assert "and the fees for that?" in llm.prompts[0]
        assert "list the credit cards" in llm.prompts[0]   # prior turn is in the prompt

    def test_a_topic_switch_resolves_to_itself(self):
        session = Session()
        session.add_turn("list the credit cards", Resolution("list the credit cards", False, "x"),
                         result())
        llm = FakeLLMClient([reply("where are the branches", used_context=False)])
        resolution = resolve_task("where are the branches", session, llm)

        assert resolution.sub_goal == "where are the branches"
        assert resolution.used_context is False

    def test_prompt_contains_the_conversation_but_not_page_text(self):
        session = Session()
        session.add_turn("list the cards", Resolution("list the cards", False, "x"), result())
        prompt = build_resolver_prompt("and the fees?", session)
        assert "Cards" in prompt and CARDS in prompt
        assert "xxxx" not in prompt and "yyyy" not in prompt


class TestResolverFailures:
    def setup_method(self):
        self.session = Session()
        self.session.add_turn("list the cards", Resolution("list the cards", False, "x"), result())

    def test_unparseable_output_falls_back_to_the_raw_task(self, caplog):
        llm = FakeLLMClient(["I think you mean the credit cards!"])
        with caplog.at_level(logging.WARNING):
            resolution = resolve_task("and the fees?", self.session, llm)
        assert resolution.sub_goal == "and the fees?"
        assert resolution.error
        assert "fell back" in caplog.text

    def test_an_empty_sub_goal_falls_back(self):
        resolution = parse_resolution(reply(""), "and the fees?")
        assert resolution.sub_goal == "and the fees?"
        assert resolution.error

    def test_an_answer_instead_of_a_rewrite_falls_back(self):
        # A model asked to rewrite a question sometimes answers it; length is
        # the giveaway, and navigating for an essay is worse than for the
        # original phrasing.
        essay = "The classic credit card has an annual fee of 300 EGP. " * 20
        resolution = parse_resolution(reply(essay), "and the fees?")
        assert resolution.sub_goal == "and the fees?"
        assert "reads as an answer" in resolution.error

    def test_an_unreachable_resolver_never_blocks_the_run(self):
        def explode(prompt):
            raise LLMError("503 UNAVAILABLE")

        resolution = resolve_task("and the fees?", self.session, FakeLLMClient(explode))
        assert resolution.sub_goal == "and the fees?"
        assert resolution.used_context is False
        assert "could not be reached" in resolution.error

    def test_markdown_fences_are_tolerated(self):
        resolution = parse_resolution(f"```json\n{reply('the classic card fees')}\n```", "x")
        assert resolution.sub_goal == "the classic card fees"

    def test_json_wrapped_in_prose_is_recovered(self):
        resolution = parse_resolution(f"Sure: {reply('the classic card fees')}", "x")
        assert resolution.sub_goal == "the classic card fees"

    def test_reasoning_is_never_empty(self):
        for raw in ["nonsense", reply("x", reasoning=""), reply("")]:
            assert parse_resolution(raw, "task").reasoning.strip()

    def test_an_empty_task_is_handled(self):
        assert resolve_task("   ", self.session, FakeLLMClient([])).sub_goal == ""


class TestFollowUpPathHint:
    def test_visited_keys_come_from_prior_turns(self):
        session = Session()
        session.add_turn("list the cards", Resolution("list the cards", False, "x"), result())
        keys = session.visited_keys()
        assert "banquemisr.com" in keys
        assert "banquemisr.com/home/smes/retail banking/pages/cards" in keys

    def test_the_bonus_ships_disabled(self):
        # Enabling it is safe -- it only reorders links discovered this run --
        # but it complicates "no pre-built index, every task navigates live
        # from the homepage" for about ten seconds of saving.
        assert config.FOLLOW_UP_PATH_BONUS == 0.0

    def test_the_bonus_only_reorders_when_enabled(self, monkeypatch):
        from agent.link_selector import Candidate, SelectionContext, score_candidate
        from browsing.extract_links import canonical_key

        url = "https://www.banquemisr.com/Home/Pages/Fees"
        link = {"label": "Fees", "url": url, "key": canonical_key(url),
                "source": "body", "is_pdf": False}
        candidate = Candidate(link=link, discovered_on=HOME, discovered_at_hop=0)
        plain = SelectionContext(hop=1, current_url=HOME)
        familiar = SelectionContext(hop=1, current_url=HOME,
                                    familiar_keys=frozenset({link["key"]}))

        assert score_candidate(candidate, familiar) == score_candidate(candidate, plain)
        monkeypatch.setattr(config, "FOLLOW_UP_PATH_BONUS", 1.5)
        assert score_candidate(candidate, familiar) > score_candidate(candidate, plain)


class TestSessionStore:
    def test_create_and_get(self):
        store = SessionStore()
        session = store.create()
        assert store.get(session.session_id) is session

    def test_unknown_id_returns_none(self):
        assert SessionStore().get("nope") is None

    def test_expired_sessions_are_swept(self):
        store = SessionStore(ttl_s=-1.0)
        session = store.create()
        assert store.get(session.session_id) is None

    def test_the_store_is_bounded(self):
        # A server left running must not accumulate sessions forever.
        store = SessionStore(max_sessions=3)
        ids = [store.create().session_id for _ in range(5)]
        assert len(store) == 3
        assert store.get(ids[0]) is None


class TestResolverVisibility:
    """The resolution must be observable -- it is the evidence that
    conversation context works, and it is what the UI shows the user."""

    def setup_method(self):
        self.session = Session()
        self.session.add_turn("list the cards", Resolution("list the cards", False, "x"), result())

    def test_the_first_turn_logs_that_it_passed_through(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.session"):
            resolve_task("find the cards", Session(), FakeLLMClient([]))
        assert "passed through" in caplog.text

    def test_a_rewrite_is_logged_with_input_and_output(self, caplog):
        llm = FakeLLMClient([reply("find the fees for the credit cards")])
        with caplog.at_level(logging.INFO, logger="agent.session"):
            resolve_task("and the fees for those?", self.session, llm)
        assert "and the fees for those?" in caplog.text
        assert "find the fees for the credit cards" in caplog.text
        assert "used_context=True" in caplog.text

    def test_a_self_contained_answer_is_logged_too(self, caplog):
        # The case that produced no log line at all before: the model decides
        # the follow-up needs no context, the UI shows nothing, and there was
        # no way to tell that from outside.
        llm = FakeLLMClient([reply("where are the branches", used_context=False)])
        with caplog.at_level(logging.INFO, logger="agent.session"):
            resolve_task("where are the branches", self.session, llm)
        assert "used_context=False" in caplog.text
        assert "changed=False" in caplog.text

    def test_a_rewrite_is_reported_even_when_the_model_denies_using_context(self, caplog):
        # used_context is a self-report; the change in text is a fact.
        llm = FakeLLMClient([reply("the fees on the credit cards", used_context=False)])
        with caplog.at_level(logging.INFO, logger="agent.session"):
            resolution = resolve_task("and the fees on those?", self.session, llm)
        assert resolution.sub_goal == "the fees on the credit cards"
        assert "changed=True" in caplog.text

    @pytest.mark.parametrize(
        "value,expected",
        [(True, True), (False, False), ("true", True), ("false", False),
         ("True", True), ("no", False), (None, False), (1, True), (0, False)],
    )
    def test_used_context_survives_a_string_boolean(self, value, expected):
        # bool("false") is True, so a model answering with the word rather than
        # the literal would have inverted the flag.
        raw = json.dumps({"sub_goal": "x", "used_context": value, "reasoning": "y"})
        assert parse_resolution(raw, "task").used_context is expected
