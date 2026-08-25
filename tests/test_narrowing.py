"""Tests for the narrowing-link rule and the deferred resolve it feeds.

The bug these exist for: a question about one product family resolved on the
category hub that merely lists that family, one hop short of the page that
answers it. Every case below is a real page from fixtures/live/.
"""

from __future__ import annotations

import pathlib

import pytest

from agent import config as agent_config
from agent.llm import FakeLLMClient
from agent import narrowing as narrowing_module
from agent.narrowing import find_narrowing_link, narrowing_terms
from agent.navigator import Navigator
from browsing.extract_links import extract_links

from tests.conftest import LIVE, live_fetcher
from tests.test_navigator import fetcher_for, page, resolve_when

CARDS = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards"
ACCOUNTS = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Accounts%20and%20Deposits"
LOANS = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Consumer%20Loans"
LIST = CARDS + "/Credit%20Cards%20List"

FIXTURES = {
    CARDS: "home-smes-retail-banking-pages-cards.html",
    ACCOUNTS: "home-smes-retail-banking-accounts-and-deposits.html",
    LOANS: "home-smes-retail-banking-consumer-loans.html",
    LIST: "home-smes-retail-banking-pages-cards-credit-cards-list.html",
}

needs_live = pytest.mark.skipif(
    not (LIVE / "manifest.json").is_file(), reason="fixtures/live/ not present"
)


def links_on(url: str) -> list[dict]:
    html = (LIVE / FIXTURES[url]).read_text(encoding="utf-8", errors="replace")
    return extract_links(html, url)


# --------------------------------------------------------------------------
# The term set
# --------------------------------------------------------------------------
class TestNarrowingTerms:
    def test_a_question_no_narrower_than_the_page_has_no_terms(self):
        assert narrowing_terms("What types of cards do you have?", CARDS) == []

    def test_the_word_the_page_lacks_is_the_term(self):
        assert narrowing_terms("What types of debit cards do you have?", CARDS) == ["debit"]

    def test_plurals_match_the_singular_in_the_path(self):
        # ".../Consumer Loans" must cover a question that says "loan".
        assert "loan" not in narrowing_terms("What are the loan terms?", LOANS)

    def test_scaffolding_is_never_a_narrowing_term(self):
        # "types", "list", "details", "overview" describe the asking, not the
        # subject; treating them as narrowing would send every question deeper.
        assert narrowing_terms("Give me the list of card types", CARDS) == []

    def test_the_planner_prefix_is_stripped(self):
        assert narrowing_terms(
            "Find information for: What types of debit cards do you have?", CARDS
        ) == ["debit"]


# --------------------------------------------------------------------------
# Picking the link
# --------------------------------------------------------------------------
@needs_live
class TestFindNarrowingLink:
    def test_finds_the_child_named_only_by_its_url(self):
        """The reported bug. Its label is the template's "More Details"."""
        hit = find_narrowing_link(
            "What types of debit cards do you have?", CARDS, links_on(CARDS)
        )
        assert hit is not None
        assert hit["url"].endswith("Debit%20Cards%20Pages")

    def test_finds_the_child_named_only_by_its_label(self):
        """The mirror image: the URL ends /Details and the label carries the name."""
        hit = find_narrowing_link(
            "What savings accounts do you have?", ACCOUNTS, links_on(ACCOUNTS)
        )
        assert hit is not None
        assert "Savings%20Accounts" in hit["url"]

    def test_a_broad_question_stays_put(self):
        for question, url in [
            ("What types of cards do you have?", CARDS),
            ("What accounts and deposits are offered?", ACCOUNTS),
            ("What consumer loans are there?", LOANS),
        ]:
            assert find_narrowing_link(question, url, links_on(url)) is None, question

    def test_abstains_when_the_site_words_it_differently(self):
        """The site says "Auto Loan"; the question says "car". No guessing."""
        assert find_narrowing_link(
            "What are the terms of the car loan?", LOANS, links_on(LOANS)
        ) is None

    def test_prefers_a_child_over_an_unrelated_page_that_shares_the_word(self):
        hit = find_narrowing_link(
            "What credit cards does Banque Misr offer?", CARDS, links_on(CARDS)
        )
        # "credit" also appears in /en/SMEs/Medium-Enterprises-Credit-Projects
        # and in the application form; the child of the page we are on wins.
        assert hit is not None
        assert hit["url"] == LIST

    def test_a_visited_page_is_not_offered_again(self):
        links = links_on(CARDS)
        visited = {link["key"] for link in links}
        assert find_narrowing_link(
            "What types of debit cards do you have?", CARDS, links, visited
        ) is None

    def test_pdfs_are_not_narrowing_links(self):
        hit = find_narrowing_link("Show the secure code guide", CARDS, links_on(CARDS))
        assert hit is None or not hit["is_pdf"]



# --------------------------------------------------------------------------
HUB = "https://www.banquemisr.com/things"
NARROW = HUB + "/blue-things"


def hub_pages(deeper_body: str = "Blue things in detail.") -> dict[str, str]:
    """A hub that lists a narrower page, in the site's own shape.

    Synthetic on purpose. fixtures/live/ holds seven English pages and none of
    the narrower ones this feature exists to reach, so the round trip has to be
    demonstrated on pages the test controls end to end.
    """
    return {
        HUB: page(f"{NARROW}|More Details", body="Every kind of thing we sell."),
        NARROW: page(body=deeper_body),
    }


def resolve_any(_sub_goal, page_dict):
    """Resolve on every page that loaded -- the hub included.

    That is the situation the feature exists for: the hub is not wrong, it is
    just less specific than what was asked.
    """
    return {"resolved": bool(page_dict["ok"]), "extracted": {"url": page_dict["url"]},
            "reason": "resolved by the test"}


def resolve_url(wanted: str):
    def validate(_sub_goal, page_dict):
        hit = page_dict.get("url") == wanted
        return {"resolved": hit, "extracted": {"url": wanted} if hit else {},
                "reason": "resolved by the test" if hit else "not this page"}
    return validate


def deferring_navigator(pages: dict[str, str], validate_fn):
    # No scripted replies: a model call would raise. That is the assertion --
    # the deeper hop is read off the URL structure, never asked for.
    return Navigator(
        FakeLLMClient([]), fetcher=fetcher_for(pages), validate_fn=validate_fn,
        seed_url=HUB,
    )


class TestDeferredResolve:
    def test_the_deeper_page_wins_when_it_also_resolves(self):
        result = deferring_navigator(hub_pages(), resolve_any).navigate(
            "what blue things do you have"
        )
        assert result.status == "resolved"
        assert result.page["url"] == NARROW
        assert result.pages_fetched == 2

    def test_the_held_verdict_comes_back_when_the_deeper_page_does_not_resolve(self):
        """The safety property: deferring can cost a fetch, never the answer."""
        pages = hub_pages(deeper_body="This page is about something else entirely.")
        result = deferring_navigator(pages, resolve_url(HUB)).navigate(
            "what blue things do you have"
        )
        assert result.status == "resolved"
        assert result.page["url"] == HUB
        assert result.extracted == {"url": HUB}
        # Exactly one extra fetch, and then it stops rather than searching on.
        assert result.pages_fetched == 2

    def test_the_held_verdict_survives_a_dead_link(self):
        """A 404 one hop down must not turn a good answer into a bad ending."""
        pages = {HUB: page(f"{NARROW}|More Details", body="Every kind of thing.")}
        result = deferring_navigator(pages, resolve_url(HUB)).navigate(
            "what blue things do you have"
        )
        assert result.status == "resolved"
        assert result.page["url"] == HUB

    def test_a_broad_question_never_pays_the_extra_fetch(self):
        result = deferring_navigator(hub_pages(), resolve_any).navigate(
            "what things do you have"
        )
        assert result.status == "resolved"
        assert result.page["url"] == HUB
        assert result.pages_fetched == 1

    def test_the_flag_restores_the_old_behaviour(self, monkeypatch):
        monkeypatch.setattr(agent_config, "DEEPEN_ON_NARROWING", False)
        result = deferring_navigator(hub_pages(), resolve_any).navigate(
            "what blue things do you have"
        )
        assert result.page["url"] == HUB
        assert result.pages_fetched == 1

    def test_only_one_deferral_per_sub_goal(self):
        """Every page resolves, so an uncapped rule would keep digging."""
        deepest = NARROW + "/small-blue-things"
        pages = hub_pages()
        pages[NARROW] = page(f"{deepest}|More Details", body="Blue things.")
        pages[deepest] = page(body="Small blue things.")
        result = deferring_navigator(pages, resolve_any).navigate(
            "what small blue things do you have"
        )
        assert result.status == "resolved"
        assert result.pages_fetched <= 1 + agent_config.MAX_NARROWING_DEFERRALS


@needs_live
class TestDeferredResolveOnRealPages:
    def test_the_hub_defers_to_the_page_named_in_the_question(self):
        """The reported shape of the bug, on the real saved pages.

        The cards hub resolves -- it does carry substance -- but the question
        named one family listed on it, and that family has its own page.
        """
        result = Navigator(
            FakeLLMClient([]), fetcher=live_fetcher(), seed_url=CARDS,
            validate_fn=resolve_when("cards"),
        ).navigate("what credit cards do you have")
        assert result.status == "resolved"
        assert result.page["url"] == LIST

# --------------------------------------------------------------------------
# The constraint that outlives this feature
# --------------------------------------------------------------------------
def _code_only(source: str) -> str:
    """The module with every comment and string literal removed.

    Prose may name the pages this was debugged against; the rule itself may
    not, and the difference is the whole guarantee.
    """
    import io, tokenize

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def test_the_rule_carries_no_product_vocabulary():
    """Delete "card" from the repo and this module must still work.

    The whole design rests on comparing the question against the page's own
    URL, so any noun that appears here would be a bug, not a tuning knob.
    """
    source = pathlib.Path(narrowing_module.__file__).read_text(encoding="utf-8")
    code = _code_only(source).lower()
    for word in ("card", "loan", "account", "deposit", "debit", "credit", "saving",
                 "fee", "product", "bank"):
        assert word not in code, word
