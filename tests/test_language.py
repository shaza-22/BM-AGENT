"""Tests for bilingual operation.

The live symptom this fixes: "ازاى افتح حساب اسلامي" was understood, but every
Arabic link had already been dropped by normalize_url, so the agent could only
choose among English links and landed on generic account opening.
"""

from __future__ import annotations

import json

import pytest

from agent import config as agent_config
from agent.config import seed_for
from agent.link_selector import (
    Candidate,
    SelectionContext,
    build_prompt,
    parse_selection,
    reply_language_instruction,
    score_candidate,
    select_next_link,
)
from agent.llm import FakeLLMClient
from browsing import config as browsing_config
from browsing.extract_links import (
    canonical_key,
    extract_links,
    link_language,
    normalize_url,
    url_language,
)
from browsing.language import arabic_ratio, detect_language, script_counts

HOME = "https://www.banquemisr.com/"
AR_HOME = "https://www.banquemisr.com/?sc_lang=ar-EG"


class TestDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ازاى افتح حساب اسلامي", "ar"),
            ("ما هي رسوم البطاقة؟", "ar"),
            ("find the annual fee of the classic card", "en"),
            ("compare credit cards", "en"),
        ],
    )
    def test_pure_input(self, text, expected):
        assert detect_language(text) == expected

    def test_mixed_input_favours_arabic(self):
        # Latin brand names inside Arabic questions are common; Arabic words
        # inside English questions are rare. The threshold is asymmetric
        # because the data is.
        assert arabic_ratio("BM Wallet ازاى") == pytest.approx(4 / 12, abs=0.01)
        assert detect_language("BM Wallet ازاى") == "ar"
        assert detect_language("ما هي رسوم بطاقة Visa Classic؟") == "ar"

    def test_a_stray_arabic_word_does_not_flip_a_long_english_task(self):
        assert detect_language("compare all the credit cards and their annual fees بطاقة") == "en"

    @pytest.mark.parametrize("text", ["", "   ", "12345", "!!! ???", "٢٠٢٤"])
    def test_no_letters_falls_back_to_the_default(self, text):
        assert detect_language(text, default="en") == "en"
        assert detect_language(text, default="ar") == "ar"

    def test_digits_are_excluded_from_the_ratio(self):
        # Arabic-Indic digits say nothing about the sentence around them.
        assert script_counts("٢٠٢٤ cards") == (0, 5)

    def test_the_threshold_is_configurable(self, monkeypatch):
        monkeypatch.setattr(browsing_config, "ARABIC_DETECTION_THRESHOLD", 0.9)
        assert detect_language("BM Wallet ازاى") == "en"

    def test_presentation_forms_count_as_arabic(self):
        assert detect_language("ﺍﻟﺮﺳﻮﻡ") == "ar"


class TestScLangPreservation:
    """The load-bearing fix.

    The Arabic site is not a separate path tree: it is the same paths with
    "?sc_lang=ar-EG". Stripping that as tracking turned every Arabic URL into
    its English twin and collapsed both onto one canonical key.
    """

    def test_the_arabic_marker_survives_normalization(self):
        assert normalize_url("/?sc_lang=ar-EG", HOME, language="ar") == AR_HOME

    def test_the_default_language_marker_is_still_stripped(self):
        # Keeps English URLs deduplicating exactly as before.
        assert normalize_url("/Home/Pages/Fees?sc_lang=en", HOME, language="en") == (
            "https://www.banquemisr.com/Home/Pages/Fees"
        )

    def test_the_two_languages_do_not_share_a_canonical_key(self):
        english = normalize_url("/Home/Pages/Fees", HOME, language="any")
        arabic = normalize_url("/Home/Pages/Fees?sc_lang=ar-EG", HOME, language="any")
        assert canonical_key(english) != canonical_key(arabic)

    def test_an_unrecognised_marker_value_is_dropped(self):
        assert normalize_url("/Home/Pages/Fees?sc_lang=zz-ZZ", HOME, language="any") == (
            "https://www.banquemisr.com/Home/Pages/Fees"
        )

    def test_url_language_reads_both_marker_forms(self):
        assert url_language("https://www.banquemisr.com/Home/x?sc_lang=ar-EG") == "ar-eg"
        assert url_language("https://www.banquemisr.com/ar-eg/Home/x") == "ar-eg"
        assert url_language("https://www.banquemisr.com/Home/x") is None


class TestPerRunLanguage:
    def test_the_language_reaches_normalize_url(self):
        arabic_url = "/Home/Pages/Fees?sc_lang=ar-EG"
        assert normalize_url(arabic_url, HOME, language="en") is None
        assert normalize_url(arabic_url, HOME, language="ar") is not None
        assert normalize_url(arabic_url, HOME, language="any") is not None

    def test_none_defers_to_the_configured_default(self, monkeypatch):
        # Read at call time, not captured as a default argument.
        monkeypatch.setattr(browsing_config, "LANGUAGE", "ar")
        assert normalize_url("/Home/x?sc_lang=ar-EG", HOME) is not None
        monkeypatch.setattr(browsing_config, "LANGUAGE", "en")
        assert normalize_url("/Home/x?sc_lang=ar-EG", HOME) is None

    def test_two_languages_can_run_in_one_process(self):
        arabic_url = "/Home/Pages/Fees?sc_lang=ar-EG"
        assert normalize_url(arabic_url, HOME, language="ar")
        assert normalize_url(arabic_url, HOME, language="en") is None
        assert normalize_url(arabic_url, HOME, language="ar")   # unchanged by the call between

    def test_arabic_percent_encoded_paths_round_trip(self):
        from urllib.parse import quote

        url = "https://www.banquemisr.com/Home/" + quote("الخدمات المصرفية")
        normalized = normalize_url(url, HOME, language="ar")
        assert normalized == normalize_url(normalized, HOME, language="ar")
        assert "الخدمات المصرفية" in canonical_key(normalized)

    def test_the_arabic_seed_carries_the_marker(self):
        assert seed_for("ar") == AR_HOME
        assert seed_for("en") == agent_config.SEED_URL
        assert seed_for(None) == agent_config.SEED_URL


class TestLinkLanguage:
    def test_the_url_marker_wins(self):
        assert link_language(AR_HOME, "Home") == "ar"

    def test_an_unmarked_url_falls_back_to_the_label_script(self):
        # The site's markers are inconsistent; the label is free to read.
        assert link_language("https://www.banquemisr.com/Home/x", "الخدمات الإسلامية") == "ar"
        assert link_language("https://www.banquemisr.com/Home/x", "Islamic services") == "en"

    def test_a_label_with_no_letters_leaves_the_language_unknown(self):
        assert link_language("https://www.banquemisr.com/Home/x", "2024") is None

    def test_extract_links_reports_the_language(self):
        html = (
            '<a href="/Home/Pages/Fees">Fees and Commissions</a>'
            '<a href="/Home/Pages/Fees?sc_lang=ar-EG">الرسوم والعمولات</a>'
        )
        languages = [link["language"] for link in extract_links(html, HOME, language="any")]
        assert languages == ["en", "ar"]

    def test_rtl_marks_are_stripped_from_arabic_labels(self):
        html = '<a href="/Home/Pages/Fees">‏الرسوم والعمولات‎</a>'
        label = extract_links(html, HOME, language="ar")[0]["label"]
        assert label == "الرسوم والعمولات"
        assert not any(char in label for char in "‎‏​﻿")


class TestCrossLanguagePolicy:
    def html(self):
        return (
            '<a href="/Home/Pages/Fees">Fees and Commissions</a>'
            '<a href="/Home/Pages/Fees?sc_lang=ar-EG">الرسوم والعمولات</a>'
            '<a href="/Home/Pages/Islamic">الخدمات المصرفية الإسلامية</a>'
        )

    def test_permissive_keeps_the_other_language(self):
        labels = [link["label"] for link in extract_links(self.html(), HOME, language="any")]
        assert len(labels) == 3

    def test_strict_drops_it(self):
        arabic = extract_links(self.html(), HOME, language="ar")
        assert all(link["language"] == "ar" for link in arabic)
        assert len(arabic) == 2

    def test_strict_uses_the_label_fallback_on_unmarked_urls(self):
        english = extract_links(self.html(), HOME, language="en")
        # /Home/Pages/Islamic carries no marker, but its label is Arabic.
        assert [link["label"] for link in english] == ["Fees and Commissions"]

    def test_the_other_language_is_ranked_below_but_still_offered(self):
        # The point of permissive: content that exists in one language only
        # must stay reachable, or the agent reports "not on the website" for
        # something that is right there.
        context = SelectionContext(hop=1, current_url=HOME, language="ar")
        same = self.candidate("/Home/a", "الرسوم", "ar")
        other = self.candidate("/Home/b", "Fees", "en")
        assert score_candidate(same, context) > score_candidate(other, context)

        from agent.link_selector import rank_candidates

        assert other in rank_candidates([same, other], set(), context)

    def test_no_penalty_when_the_run_has_no_language(self):
        context = SelectionContext(hop=1, current_url=HOME, language=None)
        english = self.candidate("/Home/b", "Fees", "en")
        assert score_candidate(english, context) == score_candidate(
            self.candidate("/Home/c", "الرسوم", "ar"), context
        )

    def test_the_policy_and_penalty_are_configurable(self):
        assert agent_config.CROSS_LANGUAGE_POLICY in ("penalise", "strict")
        assert agent_config.PENALTY_OTHER_LANGUAGE > 0

    def test_the_prompt_flags_an_other_language_candidate(self):
        context = SelectionContext(hop=0, current_url=HOME, language="ar")
        prompt = build_prompt("goal", [self.candidate("/Home/b", "Fees", "en")], context)
        assert "[en]" in prompt

    @staticmethod
    def candidate(path, label, language):
        url = f"https://www.banquemisr.com{path}"
        return Candidate(
            link={"label": label, "url": url, "key": canonical_key(url),
                  "source": "body", "is_pdf": False, "language": language},
            discovered_on=HOME, discovered_at_hop=1,
        )


class TestReplyLanguage:
    def test_the_instruction_names_the_task_language(self):
        assert "Arabic" in reply_language_instruction("ar")
        assert "English" in reply_language_instruction("en")
        assert reply_language_instruction(None) == ""

    def test_the_json_structure_is_declared_off_limits(self):
        instruction = reply_language_instruction("ar")
        assert "field names" in instruction and "outcome" in instruction

    def test_an_arabic_reply_parses(self):
        # Only the free text changes language; the keys and the enum stay English.
        raw = json.dumps(
            {"choice": 0, "outcome": "follow",
             "reasoning": "هذا الرابط يؤدي إلى صفحة الخدمات المصرفية الإسلامية",
             "confidence": 0.9},
            ensure_ascii=False,
        )
        candidates = [TestCrossLanguagePolicy.candidate("/Home/a", "الخدمات", "ar")]
        selection = parse_selection(raw, candidates, available=1)
        assert selection.url is not None
        assert selection.outcome == "follow"
        assert "الخدمات المصرفية الإسلامية" in selection.reasoning

    def test_an_arabic_no_candidate_reply_parses(self):
        raw = json.dumps(
            {"choice": -1, "outcome": "arrived", "reasoning": "وصلنا إلى الصفحة المطلوبة",
             "confidence": 0.8},
            ensure_ascii=False,
        )
        selection = parse_selection(raw, [], available=0)
        assert selection.outcome == "arrived"
        assert selection.reasoning == "وصلنا إلى الصفحة المطلوبة"

    def test_the_instruction_reaches_the_model(self):
        llm = FakeLLMClient([json.dumps({"choice": -1, "outcome": "none",
                                         "reasoning": "لا شيء", "confidence": 0.5})])
        context = SelectionContext(hop=0, current_url=HOME, language="ar")
        select_next_link(
            "goal", [TestCrossLanguagePolicy.candidate("/Home/a", "الخدمات", "ar")],
            set(), context, llm=llm,
        )
        assert "Arabic" in llm.systems[0]


class TestTrackingParams:
    @pytest.mark.parametrize(
        "param",
        ["utm_id", "utm_source", "utm_source_platform", "utm_creative_format",
         "utm_marketing_tactic", "gclid", "fbclid", "msclkid"],
    )
    def test_tracking_params_are_stripped(self, param):
        url = f"https://digital.banquemisr.com/apply?{param}=x&keep=1"
        assert normalize_url(url, HOME, language="any") == (
            "https://digital.banquemisr.com/apply?keep=1"
        )

    def test_the_reported_live_url_dedupes(self):
        messy = ("https://digital.banquemisr.com/retail-customers/account-opening/apply"
                 "?utm_id=Retail+onboarding")
        clean = "https://digital.banquemisr.com/retail-customers/account-opening/apply"
        assert canonical_key(normalize_url(messy, HOME, language="any")) == (
            canonical_key(normalize_url(clean, HOME, language="any"))
        )

    def test_ref_is_left_alone(self):
        # It appears on saved pages and some sites route on it; stripping it
        # could merge genuinely different pages.
        url = "https://www.banquemisr.com/Home/x?ref=footer"
        assert "ref=footer" in normalize_url(url, HOME, language="any")


class TestAgentMessages:
    """The code's own sentences follow the run's language too.

    A run that explains its successes in Arabic and its failures in English is
    worse than one that is consistent -- and failures are where the user most
    needs to understand what happened.
    """

    def test_the_agent_speaks_arabic_when_it_runs_out_of_links(self):

        context = SelectionContext(hop=1, current_url=HOME, language="ar")
        selection = select_next_link("goal", [], set(), context, llm=FakeLLMClient([]))
        assert detect_language(selection.reasoning) == "ar"

    def test_the_same_case_in_english(self):
        context = SelectionContext(hop=1, current_url=HOME, language="en")
        selection = select_next_link("goal", [], set(), context, llm=FakeLLMClient([]))
        assert detect_language(selection.reasoning) == "en"

    def test_a_parse_failure_is_explained_in_arabic(self):
        selection = parse_selection("not json at all", [], available=0, language="ar")
        assert detect_language(selection.reasoning) == "ar"
        assert selection.reasoning.strip()

    def test_an_unknown_language_falls_back_to_english(self):
        from agent.messages import message

        assert message("no_links_left", "fr") == message("no_links_left", "en")

    def test_an_unknown_key_never_raises(self):
        from agent.messages import message

        assert message("no_such_key", "ar") == "no_such_key"

    def test_every_message_exists_in_both_languages(self):
        from agent.messages import MESSAGES

        for key, variants in MESSAGES.items():
            assert set(variants) >= {"en", "ar"}, key
            assert detect_language(variants["ar"]) == "ar", key

    def test_cap_messages_carry_the_number(self):
        from agent.messages import message

        assert "5" in message("cap_hops", "ar", cap=5)
        assert "15" in message("cap_pages", "en", cap=15)
