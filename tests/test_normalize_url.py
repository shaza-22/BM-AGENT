"""Tests for URL normalisation and canonical keying.

These guard the two properties the navigation loop depends on: a normalised URL
must still be fetchable, and two spellings of one page must share one key.
"""

from __future__ import annotations

import pytest

from browsing import config
from browsing.extract_links import canonical_key, normalize_url

BASE = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards"
HOME = "https://www.banquemisr.com/"


class TestRejection:
    @pytest.mark.parametrize(
        "href",
        [
            "mailto:info@banquemisr.com",
            "tel:+20219888",
            "javascript:void(0)",
            "data:image/png;base64,iVBORw0KGgo=",
            "",
            "   ",
            "#main-content",
        ],
    )
    def test_unusable_schemes_and_fragments(self, href):
        assert normalize_url(href, HOME) is None

    @pytest.mark.parametrize(
        "href",
        [
            "https://www.facebook.com/BanqueMisr",
            "https://google.com",
            # A plain endswith("banquemisr.com") check would accept both of these.
            "https://evil-banquemisr.com/phish",
            "https://banquemisr.com.attacker.net/login",
        ],
    )
    def test_off_domain_rejected(self, href):
        assert normalize_url(href, HOME) is None

    def test_legitimate_subdomain_allowed(self):
        assert (
            normalize_url("https://digital.banquemisr.com/onboarding", HOME)
            == "https://digital.banquemisr.com/onboarding"
        )

    @pytest.mark.parametrize(
        "href",
        [
            "/assets/logo.png",
            "/assets/hero.JPG",
            "/styles/site.css",
            "/js/app.js",
            "/icons/favicon.ico",
            "/img/chart.svg",
        ],
    )
    def test_static_assets_rejected(self, href):
        assert normalize_url(href, HOME) is None


class TestCleaning:
    def test_csrt_token_stripped(self):
        # Same page, new token every request. Left in, the visited-set never
        # dedupes and the page budget goes on re-fetches.
        url = normalize_url("/Home/Pages/Fees?csrt=8827361882736", HOME)
        assert url == "https://www.banquemisr.com/Home/Pages/Fees"

    def test_sitecore_params_stripped(self):
        url = normalize_url(
            "/Home/Pages/Fees?sc_site=BM&sc_mode=normal&sc_itemid=%7B123%7D", HOME
        )
        assert url == "https://www.banquemisr.com/Home/Pages/Fees"

    def test_meaningful_params_survive(self):
        url = normalize_url("/Home/Pages/Search?q=tariff&csrt=abc", HOME)
        assert url == "https://www.banquemisr.com/Home/Pages/Search?q=tariff"

    def test_fragment_stripped(self):
        url = normalize_url("/Home/Pages/Fees#tariff-table", HOME)
        assert url == "https://www.banquemisr.com/Home/Pages/Fees"

    def test_literal_spaces_encoded(self):
        url = normalize_url("/Home/SMEs/Retail Banking/Consumer Loans", HOME)
        assert url == "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Consumer%20Loans"

    def test_encoded_spaces_preserved_not_doubled(self):
        url = normalize_url("/Home/SMEs/Retail%20Banking", HOME)
        assert url == "https://www.banquemisr.com/Home/SMEs/Retail%20Banking"
        assert "%2520" not in url

    def test_path_case_is_preserved(self):
        # Sitecore paths can be case-sensitive; lowercasing here would 404.
        url = normalize_url("/Home/SMEs/Retail%20Banking", HOME)
        assert "/Home/SMEs/" in url

    def test_host_is_lowercased(self):
        assert normalize_url("https://WWW.BanqueMisr.COM/Home", HOME).startswith(
            "https://www.banquemisr.com/"
        )

    def test_trailing_slash_and_duplicate_slashes_normalised(self):
        assert normalize_url("/Home//Pages///Fees/", HOME) == (
            "https://www.banquemisr.com/Home/Pages/Fees"
        )

    def test_relative_resolves_against_base(self):
        assert normalize_url("Credit%20Cards%20List", BASE + "/") == (
            "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards/"
            "Credit%20Cards%20List"
        )

    def test_protocol_relative_resolves(self):
        assert normalize_url("//www.banquemisr.com/Home/Pages/Fees", HOME) == (
            "https://www.banquemisr.com/Home/Pages/Fees"
        )


class TestLanguageFilter:
    @pytest.mark.parametrize(
        "href",
        [
            "/ar-eg/Home/Pages/Default",
            "/Home/Pages/Default?sc_lang=ar-EG",
            "/AR-EG/Home/Pages/Cards",
        ],
    )
    def test_arabic_rejected_when_language_is_en(self, href):
        assert normalize_url(href, HOME) is None

    @pytest.mark.parametrize(
        "href",
        [
            "/en/Digital-Services/Apple-pay",
            "/Home/SMEs/Retail%20Banking",  # unmarked URLs default to English
            "/Home/Pages/Fees?sc_lang=en",
        ],
    )
    def test_english_and_unmarked_accepted(self, href):
        assert normalize_url(href, HOME) is not None

    def test_language_param_stripped_after_the_check(self):
        assert normalize_url("/Home/Pages/Fees?sc_lang=en", HOME) == (
            "https://www.banquemisr.com/Home/Pages/Fees"
        )

    def test_filter_is_configurable_not_hardcoded(self, monkeypatch):
        monkeypatch.setattr(config, "LANGUAGE", "ar")
        assert normalize_url("/ar-eg/Home/Pages/Default", HOME) is not None
        assert normalize_url("/en/Digital-Services/Apple-pay", HOME) is None

    def test_filter_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "LANGUAGE", None)
        assert normalize_url("/ar-eg/Home/Pages/Default", HOME) is not None

    def test_content_segment_is_not_mistaken_for_a_language_tag(self):
        # A bare two-letter regex would read "SMEs"-style folders as languages.
        assert normalize_url("/Home/SMEs/Pages/Cards", HOME) is not None


class TestCanonicalKey:
    def test_spelling_variants_share_one_key(self):
        variants = [
            "https://www.banquemisr.com/Home/SMEs/Retail%20Banking",
            "https://www.banquemisr.com/home/smes/retail-banking",
            "https://banquemisr.com/Home/SMEs/Retail_Banking/",
            "https://www.banquemisr.com/Home/SMEs/Retail Banking",
        ]
        keys = {canonical_key(normalize_url(v, HOME)) for v in variants}
        assert len(keys) == 1

    def test_token_variants_share_one_key(self):
        first = canonical_key(normalize_url("/Home/Pages/Fees?csrt=aaa", HOME))
        second = canonical_key(normalize_url("/Home/Pages/Fees?csrt=bbb", HOME))
        assert first == second

    def test_different_pages_keep_different_keys(self):
        first = canonical_key(normalize_url("/Home/Pages/Fees", HOME))
        second = canonical_key(normalize_url("/Home/Pages/Branches", HOME))
        assert first != second

    def test_query_order_does_not_matter(self):
        first = canonical_key(normalize_url("/Home/Pages/Search?b=2&a=1", HOME))
        second = canonical_key(normalize_url("/Home/Pages/Search?a=1&b=2", HOME))
        assert first == second
