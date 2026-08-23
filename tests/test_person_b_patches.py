"""Regression tests for the patches applied to the vendored Person B package.

These live outside ``tests/person_b_vendor/`` on purpose. That directory is a
verbatim copy of their suite and gets replaced wholesale when they ship a new
drop; these tests must survive that, because their job is to fail loudly if a
re-vendor silently drops a patch. Each test names the patch it guards.

The bugs are recorded, with measurements, in ``src/person_b/PATCHES.md``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from person_b.api import (  # noqa: E402
    finalize,
    next_pending_sub_goal,
    plan_task,
    validate,
    validate_answer,
)
from person_b.planning.planner import _extract_requested_fields  # noqa: E402
from person_b.validation import validator  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "live"

ACCOUNTS = "home-smes-retail-banking-accounts-and-deposits"
LOANS = "home-smes-retail-banking-consumer-loans"
CARDS = "home-smes-retail-banking-pages-cards"
CARD_LIST = "home-smes-retail-banking-pages-cards-credit-cards-list"
CLASSIC = "home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards"
HOME = "index"


def page(stem: str) -> str:
    return (FIXTURES / f"{stem}.txt").read_text(encoding="utf-8", errors="replace")


def verdict_for(task: str, stem: str) -> dict:
    """Run a user task through Person B exactly as agent/loop.py does."""
    sub_goal = next_pending_sub_goal(plan_task(task))
    return validate(sub_goal, page(stem), source_url=f"https://www.banquemisr.com/{stem}")


# --------------------------------------------------------------------------
# PATCH 6 + 7 -- category rubber-stamping
# --------------------------------------------------------------------------
class TestCategoryRubberStamping:
    """A page in the right category must not resolve any question in it.

    Before the patch, every one of these resolved against the accounts hub with
    the reason "Found Accounts and Deposits offerings and details." -- the
    branch tested category membership and never the question's subject.
    """

    # The three probe queries from the audit, kept verbatim.
    @pytest.mark.parametrize(
        "task",
        [
            "Can I open a joint account with my dog?",
            "What is the interest rate on a Martian savings account?",
            "Does Banque Misr offer a cryptocurrency deposit account?",
        ],
    )
    def test_absurd_in_category_questions_must_not_resolve(self, task: str) -> None:
        result = verdict_for(task, ACCOUNTS)
        assert result["resolved"] is False, (
            f"{task!r} resolved against the accounts hub; the page says nothing "
            f"about its subject. reason={result.get('reason')!r}"
        )

    @pytest.mark.parametrize(
        "task,stem",
        [
            ("Does Banque Misr offer student accounts?", ACCOUNTS),
            ("How do I open an Islamic account?", ACCOUNTS),
            ("What is the interest rate on a car loan?", LOANS),
        ],
    )
    def test_plausible_but_absent_subjects_must_not_resolve(self, task: str, stem: str) -> None:
        # These read like real questions, which is what makes them the dangerous
        # case: "student", "Islamic" and "car" appear nowhere in these pages.
        assert verdict_for(task, stem)["resolved"] is False

    def test_discovery_branch_checks_the_question(self) -> None:
        # The discovery branch resolved on ">= 5 entities of any kind", and every
        # content page has 6-12, so any discovery question resolved anywhere.
        assert verdict_for("What credit cards does Banque Misr offer?", ACCOUNTS)["resolved"] is False

    def test_the_right_page_still_resolves(self) -> None:
        # The gate must not have simply turned everything off.
        assert verdict_for("What credit cards does Banque Misr offer?", CARD_LIST)["resolved"] is True
        assert verdict_for("What personal loans are available?", LOANS)["resolved"] is True
        assert verdict_for("What accounts and deposits does the bank offer?", ACCOUNTS)["resolved"] is True


# --------------------------------------------------------------------------
# PATCH 5 -- boilerplate false positive
# --------------------------------------------------------------------------
class TestBoilerplateFalsePositive:
    """The homepage names every product in its nav and answers none of them.

    This is the most damaging of the three: resolving on the seed means
    navigation stops at hop 0 and the run reports success having fetched one
    page. The project's claim is that it navigates live from the homepage.
    """

    @pytest.mark.parametrize(
        "task",
        [
            "How do I open an account?",
            "Tell me about Islamic banking accounts",
            "What credit cards does Banque Misr offer?",
            "What personal loans are available?",
            "Where is the schedule of fees and commissions?",
        ],
    )
    def test_homepage_never_resolves(self, task: str) -> None:
        result = verdict_for(task, HOME)
        assert result["resolved"] is False, (
            f"{task!r} resolved on the seed page. reason={result.get('reason')!r}"
        )

    def test_overview_requires_page_substance_not_just_evidence(self) -> None:
        # Root cause: _evaluate_field_coverage returned ("overview" found,
        # nothing missing) without looking at the page, and the caller resolves
        # on "evidence and not missing_fields". Any page has some evidence.
        found, missing = validator._evaluate_field_coverage(["overview"], {})
        assert missing == ["overview"] and found == []

        found, missing = validator._evaluate_field_coverage(
            ["overview"], {"entities": [{"name": "Current Accounts"}]}
        )
        assert missing == [] and found == ["overview"]

    def test_a_bare_table_is_not_page_substance(self) -> None:
        # The site's currency widget parses as a table on every page, headers
        # included, so a table alone cannot stand for an overview.
        _found, missing = validator._evaluate_field_coverage(
            ["overview"],
            {"tables": [{"table_name": "Currencies and Exchange Rates",
                         "headers": ["{{fromCurrency}}", "Cash", "Transfer"]}]},
        )
        assert missing == ["overview"]


# --------------------------------------------------------------------------
# PATCH 3 + 12 -- the topic gate, and the tolerance it ships at
# --------------------------------------------------------------------------
class TestTopicGate:
    def test_scaffolding_words_are_not_subject_matter(self) -> None:
        tokens = validator._distinctive_query_tokens(
            "Tell me about the fees, would you show me any documents you have?"
        )
        assert "tell" not in tokens and "about" not in tokens and "show" not in tokens
        assert "fees" in tokens

    def test_a_question_of_pure_scaffolding_does_not_block(self) -> None:
        ok, missing = validator._question_topics_present("What is it?", "anything at all")
        assert ok is True and missing == []

    def test_shipped_tolerance_is_zero(self) -> None:
        # Chosen by the sweep in scripts/validator_bench.py: tolerance 0 is the
        # only setting with no false positives. Raising it buys back one false
        # negative and costs three. If this constant moves, re-run the sweep.
        assert validator._TOPIC_MISS_TOLERANCE == 0

    def test_gate_reads_the_body_not_the_navigation(self) -> None:
        # On an interior page the nav is stripped before matching: "islamic"
        # appears in every page's nav as "Islamic Banking", but not in the
        # accounts hub's body.
        ok, missing = validator._question_topics_present(
            "Find information for: Tell me about Islamic accounts", page(ACCOUNTS)
        )
        assert ok is False and "islamic" in missing

    def test_the_gate_alone_does_not_save_the_homepage(self) -> None:
        # Worth pinning because the gate's docstring is easy to over-read. The
        # boilerplate stripper finds the body by locating a breadcrumb, and the
        # homepage has none, so nothing is stripped there and nav words match.
        # The homepage is stopped by the substance check in PATCH 5 instead --
        # which is why both patches are needed, not either one.
        ok, _missing = validator._question_topics_present(
            "Find information for: Tell me about Islamic accounts", page(HOME)
        )
        assert ok is True
        assert verdict_for("Tell me about Islamic accounts", HOME)["resolved"] is False


# --------------------------------------------------------------------------
# PATCH 2 + 10 -- requirements the user never stated
# --------------------------------------------------------------------------
class TestInventedRequirements:
    def test_card_wording_alone_does_not_demand_fees_and_benefits(self) -> None:
        assert _extract_requested_fields("Tell me about Banque Misr payment cards") == ["overview"]

    def test_comparison_still_gets_default_axes(self) -> None:
        # Narrowed, not deleted: a comparison with no stated dimension does
        # need axes, and their test_dynamic_plan_expansion depends on these.
        assert _extract_requested_fields("Compare all Banque Misr credit cards") == ["fees", "benefits"]

    def test_asking_for_a_document_does_not_demand_eligibility(self) -> None:
        assert "eligibility" not in _extract_requested_fields("Show me the fees and rates document")
        # Questions that really are about eligibility still land there.
        assert "eligibility" in _extract_requested_fields("What are the eligibility requirements?")
        assert "eligibility" in _extract_requested_fields("What is the requirement to apply?")

    def test_an_open_question_resolves_on_the_hub_that_answers_it(self) -> None:
        assert verdict_for("Tell me about Banque Misr payment cards", CARDS)["resolved"] is True


# --------------------------------------------------------------------------
# PATCH 8 + 9 + 11 -- over-strict rejections
# --------------------------------------------------------------------------
class TestOverStrictRejections:
    def test_entity_inferred_from_a_question_drops_article_and_punctuation(self) -> None:
        # "...for the Classic credit card?" yielded the entity
        # "the Classic credit card?", which matches no page, so the page
        # holding the answer was rejected for irrelevance.
        result = verdict_for("What is the annual fee for the Classic credit card?", CLASSIC)
        assert result["resolved"] is True, result.get("reason")

    def test_field_coverage_can_be_satisfied_by_entity_names(self) -> None:
        # A hub whose answer is a set of documents scored zero coverage because
        # only table headers and sections were searched.
        found, missing = validator._evaluate_field_coverage(
            ["fees"],
            {"entities": [{"name": "Banque Misr payment cards Fees , Limits and commission"}]},
        )
        assert found == ["fees"] and missing == []

    def test_interest_rate_field_looks_for_interest_and_rates(self) -> None:
        found, _missing = validator._evaluate_field_coverage(
            ["interest_rate"],
            {"entities": [{"name": "Deposits Interest Rates for individual customers"}]},
        )
        assert found == ["interest_rate"]


# --------------------------------------------------------------------------
# PATCH 14 -- unsupported prose must not survive verification
# --------------------------------------------------------------------------
class TestUnsupportedProseIsStruck:
    VISITED = ["https://www.banquemisr.com/Home/Pages/Fees"]

    def _claims(self) -> list[dict]:
        return [
            {"id": "c1", "statement": "The issuance fee is 100 EGP.",
             "source_url": self.VISITED[0]},
            {"id": "c2", "statement": "The annual fee is 250 EGP.",
             "source_url": self.VISITED[0]},
            {"id": "c3", "statement": "The card includes free airport lounge access.",
             "source_url": "https://www.banquemisr.com/Home/Never/Visited"},
        ]

    def test_a_partly_verified_answer_loses_only_the_bad_sentence(self) -> None:
        claims = self._claims()
        draft = {"draft_answer": "\n\n".join(c["statement"] for c in claims),
                 "sources": [], "missing_info": []}
        verification = validate_answer(claims, self.VISITED, strict=True)
        final = finalize(draft, verification)

        assert "airport lounge" not in final["answer"], (
            "unsupported claim survived in the prose with verified sources beneath it"
        )
        assert "issuance fee is 100 EGP" in final["answer"]
        assert any("airport lounge" in n for n in final["not_found"])

    def test_the_struck_sentence_is_reported_not_silently_dropped(self) -> None:
        claims = self._claims()
        verification = validate_answer(claims, self.VISITED, strict=True)
        final = finalize({"draft_answer": "\n\n".join(c["statement"] for c in claims),
                          "sources": [], "missing_info": []}, verification)
        assert final["metadata"]["prose_removed"], "removal happened but was not reported"

    def test_support_rate_is_surfaced(self) -> None:
        claims = self._claims()
        verification = validate_answer(claims, self.VISITED, strict=True)
        final = finalize({"draft_answer": "x", "sources": [], "missing_info": []}, verification)
        meta = final["metadata"]
        assert 0.0 <= meta["support_rate"] <= 1.0
        assert meta["claims_total"] == 3
        assert meta["claims_supported"] + len(verification["failed"]) == 3

    def test_a_bullet_list_stripped_to_nothing_loses_its_lead_in(self) -> None:
        claims = [{"id": "c1", "statement": "Unverifiable fact one.",
                   "source_url": "https://www.banquemisr.com/Home/Never/Visited"}]
        draft = {"draft_answer": "Based on Banque Misr official documentation:\n- Unverifiable fact one.",
                 "sources": [], "missing_info": []}
        final = finalize(draft, validate_answer(claims, self.VISITED, strict=True))
        assert "Based on Banque Misr official documentation:" not in final["answer"]

    def test_a_fully_supported_answer_is_untouched(self) -> None:
        claims = self._claims()[:2]
        draft = {"draft_answer": "\n\n".join(c["statement"] for c in claims),
                 "sources": [], "missing_info": []}
        final = finalize(draft, validate_answer(claims, self.VISITED, strict=True))
        assert final["answer"] == draft["draft_answer"]
        assert final["metadata"]["support_rate"] == 1.0


# --------------------------------------------------------------------------
# PATCH 1 + 15 -- mojibake, and a category noun in the delivered answer
# --------------------------------------------------------------------------
class TestVendoredConstants:
    def test_the_mangled_arabic_marker_is_gone(self) -> None:
        from person_b.extraction import extractor

        assert not hasattr(extractor, "_NAV_START_MARKERS")
        # The string may still appear in the patch note that records why it was
        # removed; what must not survive is a live occurrence in code.
        code_lines = [
            line for line in pathlib.Path(extractor.__file__).read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        ]
        assert not any("ie ??? ????" in line for line in code_lines)

    def test_the_answer_does_not_name_a_category_it_has_not_established(self) -> None:
        from person_b.api import synthesize

        result = synthesize(
            "What personal loans are available?",
            validated_results=[{
                "source_url": "https://www.banquemisr.com/Home/Loans",
                "extracted": {"entities": [{"name": "Personal Loans"},
                                           {"name": "Car Loans"},
                                           {"name": "Mortgage"}]},
            }],
        )
        assert "credit cards" not in result["draft_answer"].lower(), result["draft_answer"]


# --------------------------------------------------------------------------
# The benchmark itself, as a test
# --------------------------------------------------------------------------
def test_validator_benchmark_has_no_false_positives() -> None:
    """Guards the headline number: 0 false positives across 23 labelled cases.

    A false positive stops navigation on a page that cannot answer the
    question, which is worse than an extra hop -- so this is the assertion that
    must not regress, and it is stated as a hard zero rather than a threshold.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import validator_bench

    report = validator_bench.run()
    assert not report["false_positives"], [
        (t, p) for t, p, _s, _n in report["false_positives"]
    ]
    # Baseline before the patches was 10/23 with 9 false positives.
    assert report["correct"] >= 21, f"{report['correct']}/{report['total']}"
