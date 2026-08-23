"""Composing the final answer, and the grounding check that guards it.

The vendored synthesis is template-based with no model in it, so it can fill
slots but cannot write prose. ``agent/answer.py`` buys the prose; everything
here is about making sure it cannot buy a hallucination with it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from agent.answer import compose_answer  # noqa: E402
from agent.grounding import check_grounding  # noqa: E402
from agent.llm import FakeLLMClient, LLMError  # noqa: E402

CLAIMS = [
    {"entity": "Issuance", "field": "Details", "value": "EGP 250",
     "statement": "Fees and charges: Issuance — EGP 250.",
     "source_url": "https://www.banquemisr.com/classic"},
    {"entity": "Renewal", "field": "Details", "value": "EGP 250",
     "statement": "Fees and charges: Renewal — EGP 250.",
     "source_url": "https://www.banquemisr.com/classic"},
    {"entity": "Supplementary cards issuance and renewal", "field": "Details", "value": "EGP100",
     "statement": "Fees and charges: Supplementary cards issuance and renewal — EGP100.",
     "source_url": "https://www.banquemisr.com/classic"},
    {"entity": "Penalty for delay", "field": "Details", "value": "EGP75",
     "statement": "Fees and charges: Penalty for delay — EGP75.",
     "source_url": "https://www.banquemisr.com/classic"},
    {"entity": "Interest rate", "field": "Details", "value": "4% monthly",
     "statement": "Fees and charges: Interest rate — 4% monthly.",
     "source_url": "https://www.banquemisr.com/classic"},
]


class TestGrounding:
    def test_a_faithful_sentence_survives(self):
        report = check_grounding("Issuance — EGP 250. Renewal — EGP 250.", CLAIMS)
        assert report.struck == []
        assert report.rate == 1.0

    def test_an_invented_figure_is_struck(self):
        """The failure that matters most: a wrong number carrying a source."""
        report = check_grounding("The annual fee is EGP 500.", CLAIMS)
        assert report.text == ""
        assert "not in the evidence" in report.struck[0][1]

    def test_a_bare_assertion_with_no_checkable_token_is_struck(self):
        """The case scaffolding must never be allowed to cover.

        It carries no figure, so nothing can be checked against the evidence --
        and it is exactly what a model reaches for when the evidence does not
        cover something. Negation disqualifies a sentence from scaffolding
        regardless of anything else about it.
        """
        report = check_grounding("The Classic card has no annual fee.", CLAIMS)
        assert report.text == ""
        assert "asserts something the evidence does not support" in report.struck[0][1]

    @pytest.mark.parametrize("sentence", [
        "There is no annual fee.",
        "The card is free for the first year.",
        "Cash withdrawals are unlimited.",
        "Banque Misr is Egypt's second largest bank.",
        "This card is not available to students.",
    ])
    def test_assertions_without_figures_are_still_struck(self, sentence: str):
        assert check_grounding(sentence, CLAIMS).text == ""

    @pytest.mark.parametrize("sentence", [
        "Here are the fees for the Classic credit card:",
        "**Card fees**",
        "## Interest and installments",
        "These figures are taken from the card's fee schedule.",
        "In summary:",
    ])
    def test_scaffolding_survives(self, sentence: str):
        """The rule used to strike every one of these for carrying no figure.

        What survived was only the bare fact sentences, so the answer read as a
        flat list. Measured on a realistic 17-sentence answer: 8 kept before,
        14 after, with the same assertions struck in both.
        """
        assert check_grounding(sentence, CLAIMS).text == sentence

    def test_grouped_structure_survives_the_round_trip(self):
        answer = (
            "Here are the fees:\n\n"
            "**Card fees**\n"
            "Issuance — EGP 250.\n"
            "Renewal — EGP 250.\n\n"
            "**Penalties**\n"
            "Penalty for delay — EGP75.\n"
        )
        report = check_grounding(answer, CLAIMS)
        assert report.struck == []
        assert "**Card fees**" in report.text
        assert report.text.count("\n\n") >= 2, "blank lines between groups were lost"

    def test_a_decimal_point_is_not_a_sentence_boundary(self):
        """"2.81%" split into "2." and "81%", and both then failed the figure
        check for numbers the model never wrote."""
        claims = [{"entity": "3", "value": "2.81%"}]
        report = check_grounding("3 — 2.81%.", claims)
        assert report.struck == [], report.struck

    def test_a_numeric_label_does_not_trigger_the_pair_rule(self):
        """A Tenor row is labelled "3"; every figure matches it."""
        claims = [{"entity": "3", "value": "2.81%"}, {"entity": "6", "value": "2.77%"}]
        assert check_grounding("6 — 2.77%.", claims).struck == []

    def test_off_evidence_prose_is_struck(self):
        report = check_grounding("Banque Misr is Egypt's second largest bank.", CLAIMS)
        assert report.text == ""

    def test_recombination_of_two_real_facts_is_struck(self):
        """Both 'Penalty for delay' and 'EGP 250' are real; the pairing is not."""
        report = check_grounding("Penalty for delay is EGP 250.", CLAIMS)
        assert report.text == ""
        assert "is not its value" in report.struck[0][1]

    def test_a_nested_label_does_not_cause_a_false_strike(self):
        """'Issuance' is a substring of 'Supplementary cards issuance and renewal'.

        Checking every matching label would strike this correct sentence for
        failing to quote the shorter label's value, so only the longest label
        present is checked.
        """
        report = check_grounding("Supplementary cards issuance and renewal — EGP100.", CLAIMS)
        assert report.struck == [], report.struck

    def test_a_mixed_answer_keeps_what_is_grounded(self):
        report = check_grounding(
            "Issuance — EGP 250. It also includes airport lounge access worth EGP 900.", CLAIMS)
        assert "EGP 250" in report.text
        assert "900" not in report.text
        assert report.sentences_total == 2 and len(report.struck) == 1

    def test_thousands_separators_do_not_break_a_match(self):
        claims = [{"entity": "Monthly limit", "value": "10,000 EGP Monthly"}]
        assert check_grounding("Monthly limit — 10,000 EGP Monthly.", claims).struck == []

    def test_with_no_claims_everything_is_struck(self):
        """Never a rubber stamp in the case where there is least to go on."""
        report = check_grounding("Anything at all, stated confidently.", [])
        assert report.text == ""
        assert "no claims" in report.struck[0][1]


class TestTheHardGate:
    def test_no_claims_means_no_model_call(self):
        """The largest hallucination risk in generating an answer at all.

        With nothing to ground on, a model asked about a bank writes from what
        it happens to know, and every word of that inherits the interface's
        credibility. This is a guard clause, not a prompt instruction.
        """
        called = []

        def spy(prompt: str) -> str:
            called.append(prompt)
            return "Banque Misr was founded in 1920 by Talaat Harb."

        result = compose_answer("What are the fees?", [], FakeLLMClient(spy))
        assert called == [], "the model was invited to write with no evidence"
        assert result.attempted is False
        assert result.used is False
        assert result.text == ""
        assert "refusing to generate" in result.reason


class TestComposition:
    def test_a_grounded_answer_is_used(self):
        llm = FakeLLMClient(lambda p: "Issuance — EGP 250. Renewal — EGP 250.")
        result = compose_answer("What are the fees on the Classic card?", CLAIMS, llm)
        assert result.used is True
        assert "EGP 250" in result.text

    def test_an_ungrounded_answer_is_discarded_whole(self):
        llm = FakeLLMClient(lambda p: "The card is free for the first year.")
        result = compose_answer("What are the fees?", CLAIMS, llm)
        assert result.attempted is True
        assert result.used is False
        assert "failed grounding" in result.reason

    def test_a_model_failure_never_costs_the_answer(self):
        """Composition improves wording; it must not be able to remove one."""
        def dies(_p: str) -> str:
            raise LLMError("quota exhausted for today")

        result = compose_answer("What are the fees?", CLAIMS, FakeLLMClient(dies))
        assert result.attempted is True and result.used is False
        assert "model unavailable" in result.reason

    def test_the_prompt_carries_the_facts_and_forbids_inventing_figures(self):
        seen = {}

        def capture(prompt: str) -> str:
            seen["prompt"] = prompt
            return "Issuance — EGP 250."

        compose_answer("What are the fees?", CLAIMS, FakeLLMClient(capture))
        assert "EGP 250" in seen["prompt"]
        assert "Penalty for delay" in seen["prompt"]
        assert "only" in seen["prompt"].lower()

    def test_only_claims_offered_to_the_model_are_used_for_grounding(self):
        """Grounding must judge against the same set the model was shown.

        Otherwise a figure the model was never given could still pass, or one
        it was given could be struck -- either way the check would be measuring
        something other than what happened.
        """
        llm = FakeLLMClient(lambda p: "Interest rate — 4% monthly.")
        result = compose_answer("What is the rate?", CLAIMS, llm, max_facts=2)
        assert result.claims_offered == 2
        # "4% monthly" is claim 5, outside the offered set, so it is not grounded.
        assert result.used is False


class TestAttributionIsTextual:
    """PATCH 17: a claim must be *said* by the page it cites, not merely cite it."""

    URL = "https://www.banquemisr.com/classic"
    PAGE = "Fees and charges\nIssuance EGP 250\nRenewal EGP 250\nInterest rate 4% monthly\n"

    def _verify(self, claims, visited):
        from person_b.api import validate_answer

        return validate_answer(claims, visited, strict=True)

    def test_a_claim_the_page_does_not_make_is_rejected(self):
        forged = [{"id": "c1", "statement": "The annual fee is EGP 9999.",
                   "value": "EGP 9999", "source_url": self.URL}]
        result = self._verify(forged, [{"url": self.URL, "text": self.PAGE}])
        assert result["all_supported"] is False
        assert "does not appear" in result["failed"][0]["reason"]

    def test_a_claim_the_page_does_make_is_accepted(self):
        real = [{"id": "c1", "statement": "Issuance — EGP 250.",
                 "value": "EGP 250", "source_url": self.URL}]
        result = self._verify(real, [{"url": self.URL, "text": self.PAGE}])
        assert result["all_supported"] is True

    def test_bare_urls_fall_back_to_the_old_url_rule(self):
        """Callers that cannot supply page text keep the previous behaviour.

        Their own suite passes URL strings, so the patch has to be strictly
        stronger where text is available and identical where it is not.
        """
        forged = [{"id": "c1", "statement": "The annual fee is EGP 9999.",
                   "value": "EGP 9999", "source_url": self.URL}]
        assert self._verify(forged, [self.URL])["all_supported"] is True

    def test_an_unvisited_page_is_still_rejected(self):
        real = [{"id": "c1", "statement": "Issuance — EGP 250.",
                 "value": "EGP 250", "source_url": "https://www.banquemisr.com/never"}]
        result = self._verify(real, [{"url": self.URL, "text": self.PAGE}])
        assert result["all_supported"] is False


class TestClaimsCarryValues:
    """PATCH 16: a table row is a label and a value, not a cell to iterate."""

    def test_a_fee_row_becomes_a_claim_with_its_amount(self):
        from person_b.api import synthesize

        result = synthesize("fees", validated_results=[{
            "source_url": "https://www.banquemisr.com/classic",
            "extracted": {"tables": [{
                "table_name": "Fees and charges",
                "headers": ["Fees and charges", "Details"],
                "records": [{"Fees and charges": "Issuance", "Details": "EGP 250"}],
            }]},
        }])
        statements = [c["statement"] for c in result["claims"]]
        assert any("EGP 250" in s for s in statements), statements
        assert not any(s == "Fees and charges: Fees and charges is Issuance." for s in statements)
        assert result["claims"][0]["value"] == "EGP 250"

    def test_the_values_reach_the_answer_text(self):
        """They were added to claims and never to the prose."""
        from person_b.api import synthesize

        result = synthesize("fees", validated_results=[{
            "source_url": "https://www.banquemisr.com/classic",
            "extracted": {
                "entities": [{"name": f"Card {i}"} for i in range(4)],
                "tables": [{
                    "table_name": "Fees and charges",
                    "headers": ["Fees and charges", "Details"],
                    "records": [{"Fees and charges": "Issuance", "Details": "EGP 250"}],
                }],
            },
        }])
        assert "EGP 250" in result["draft_answer"], result["draft_answer"]

    def test_a_single_column_record_is_a_field_value_pair(self):
        from person_b.reasoning.synthesize import _row_facts

        facts = _row_facts({"records": [{"Issuance": "250 EGP"}]})
        assert facts == [{"label": "Issuance", "field": "Issuance", "value": "250 EGP"}]


class TestTheAbbreviationSplitBug:
    """Two table rows merged in a generated answer, with one label beheaded.

    Reported from a live run:

        "Reissuance of are placement for lost or damaged cards: EGP100
         S and other banks' machines within Egypt: 2% of the amount withdrawn
         with minimum EGP15"

    "Cash withdrawals through BM ATMs and P.O." had gone missing and the
    remainder had attached itself to the previous row's value.

    The extraction was innocent -- these are two clean rows all the way through
    Person B's table parser. The damage was done here: the sentence splitter
    treated every "." as a boundary, so "P.O.S" became three fragments, the two
    that carried no figure were struck, and the survivors were rejoined with a
    space straight onto the previous line.
    """

    ROW_LABEL = "Cash withdrawals through BM ATMs and P.O.S and other banks’ machines within Egypt"
    ROW_VALUE = "2% of the amount withdrawn with minimum EGP15"
    PREV_LABEL = "Reissuance of are placement for lost or damaged cards"

    FIXTURE = ("fixtures/live/home-smes-retail-banking-pages-cards-credit-cards-"
               "pages-classic-credit-cards.txt")

    def test_the_fixture_still_holds_the_row_intact(self):
        """Guards the extraction side, which was suspected and is innocent."""
        from person_b.extraction.extractor import extract_content

        raw = (pathlib.Path(__file__).resolve().parent.parent / self.FIXTURE).read_text(
            encoding="utf-8", errors="replace")
        tables = extract_content(raw, ["fees"]).extracted.get("tables", [])
        rows = [rec for t in tables for rec in (t.get("records") or [])]
        labels = [str(list(r.values())[0]) for r in rows if r]
        assert self.ROW_LABEL in labels, "the row is no longer parsed as one row"
        assert self.PREV_LABEL in labels

    def test_an_abbreviation_does_not_split_a_sentence(self):
        from agent.grounding import _split_sentences

        parts = _split_sentences(f"{self.ROW_LABEL}: {self.ROW_VALUE}")
        assert len(parts) == 1, parts
        assert "P.O.S" in parts[0]

    def test_the_two_rows_survive_grounding_separately(self):
        from agent.grounding import check_grounding

        claims = [
            {"entity": self.PREV_LABEL, "value": "EGP100"},
            {"entity": self.ROW_LABEL, "value": self.ROW_VALUE},
        ]
        text = f"{self.PREV_LABEL}: EGP100\n{self.ROW_LABEL}: {self.ROW_VALUE}"
        report = check_grounding(text, claims)

        assert report.struck == [], report.struck
        assert self.ROW_LABEL in report.text, "the label was beheaded again"
        # The exact corruption: the tail of one row welded onto the previous.
        assert "EGP100 S and other banks" not in report.text

    def test_other_abbreviations_survive_too(self):
        from agent.grounding import _split_sentences

        for text in ["Available at the U.S. branch.", "Contact Mr. Hassan for details.",
                     "Open 9 a.m. to 5 p.m. daily."]:
            assert len(_split_sentences(text)) == 1, text


class TestInternalIdentifiersNeverReachTheReader:
    """A fact's label is ``entity``, falling back to ``field``.

    ``field`` holds programmatic names. When PATCH 15 renamed the entity-list
    claim's field to ``entity_list`` without giving it an ``entity``, the
    fallback put that identifier in front of the model -- which has no way to
    know it is not what the bank calls the thing, and copied it into the answer.
    A live recording came back reading "entity_list — Credit Card, Debit
    cards, …".
    """

    def test_the_entity_list_claim_carries_a_human_label(self):
        from person_b.api import synthesize

        result = synthesize("what cards are there", validated_results=[{
            "source_url": "https://www.banquemisr.com/cards",
            "extracted": {"entities": [{"name": "Credit Card"}, {"name": "Debit cards"},
                                       {"name": "Salaries Cards"}]},
        }])
        claim = result["claims"][0]
        assert claim["entity"], "no human label, so the composer falls back to `field`"
        assert "_" not in claim["entity"]

    def test_the_model_is_never_shown_an_identifier(self):
        from agent.answer import _facts_block

        block = _facts_block([{"field": "entity_list", "value": "Credit Card, Debit cards",
                               "source_url": "https://x/y"}], 5)
        assert "entity_list" not in block
        assert "Entity list" in block

    def test_a_future_internal_field_is_humanised_too(self):
        from agent.answer import _humanise_label

        assert _humanise_label("some_future_field") == "Some future field"
        assert _humanise_label("Issuance") == "Issuance"
        assert _humanise_label("Supplementary cards issuance and renewal") == \
            "Supplementary cards issuance and renewal"

    def test_a_real_label_containing_no_underscore_is_left_alone(self):
        from agent.answer import _humanise_label

        for label in ["Interest rate", "Penalty for delay", "3", ""]:
            assert _humanise_label(label) == label
