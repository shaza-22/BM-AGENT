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
        report = check_grounding("The Classic card has no annual fee.", CLAIMS)
        assert report.text == ""
        assert "no label or value" in report.struck[0][1]

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
