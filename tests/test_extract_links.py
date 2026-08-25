"""Tests for link extraction: the selector, the label chain, regions, dedupe.

Every HTML string here is a shape observed on the live site, reduced to the
smallest markup that reproduces it.
"""

from __future__ import annotations

from browsing import config
from browsing.extract_links import canonical_key, extract_links, is_pdf_hint, slug_label

HOME = "https://www.banquemisr.com/"


def labels(links):
    return [link["label"] for link in links]


def by_key(links, url):
    key = canonical_key(url)
    return next(link for link in links if link["key"] == key)


class TestSelector:
    def test_vue_components_carrying_href_are_found(self):
        # The main nav is not anchors. find_all("a", href=True) returns nothing
        # here, which would strand the agent on the homepage.
        html = """
        <b-nav-item-dropdown text="Digital Services">
          <b-dropdown-item href="/en/Digital-Services/Apple-pay">Apple Pay</b-dropdown-item>
        </b-nav-item-dropdown>
        """
        links = extract_links(html, HOME)
        assert labels(links) == ["Apple Pay"]
        assert links[0]["url"] == "https://www.banquemisr.com/en/Digital-Services/Apple-pay"

    def test_anchors_and_components_coexist(self):
        html = """
        <nav><b-dropdown-item href="/Home/Pages/Cards">Cards</b-dropdown-item></nav>
        <footer><a href="/Home/Pages/Fees">Fees</a></footer>
        """
        assert labels(extract_links(html, HOME)) == ["Cards", "Fees"]

    def test_stylesheet_links_are_not_navigation(self):
        html = '<link rel="stylesheet" href="/styles/site.css"><a href="/Home/Pages/Fees">Fees</a>'
        assert labels(extract_links(html, HOME)) == ["Fees"]

    def test_empty_html_returns_empty_list(self):
        assert extract_links("", HOME) == []


class TestLabelFallbackChain:
    def test_element_text_wins(self):
        html = '<a href="/Home/Pages/Fees">Fees and Commissions</a>'
        assert labels(extract_links(html, HOME)) == ["Fees and Commissions"]

    def test_heading_plus_description_is_combined(self):
        html = """
        <a href="/Home/SMEs/Retail Banking/Consumer Loans">
          <h3 class="banking-sector__title">Get a Loan</h3>
          <div class="banking-sector__brief">Get the suitable Loan from Banque Misr Loan types</div>
        </a>
        """
        assert labels(extract_links(html, HOME)) == [
            "Get a Loan Get the suitable Loan from Banque Misr Loan types"
        ]

    def test_long_label_is_capped_on_a_word_boundary(self):
        html = f'<a href="/Home/Pages/Long">{"word " * 100}</a>'
        label = extract_links(html, HOME)[0]["label"]
        assert len(label) <= 184  # cap plus the ellipsis
        assert label.endswith("...")
        assert not label.rstrip(".").endswith("wor")

    def test_image_only_link_falls_back_to_img_alt(self):
        html = """
        <a href="/Home/Pages/Payroll%20Services">
          <div class="banking-kinds__img"><img src="/tiles/payroll.png" alt="Payroll Services"/></div>
        </a>
        """
        assert labels(extract_links(html, HOME)) == ["Payroll Services"]

    def test_image_only_link_with_no_alt_falls_back_to_slug(self):
        # Sitecore paths read like prose, so the slug is a usable label.
        html = """
        <a href="/Home/Pages/Banque Misr visa business cards">
          <div class="banking-kinds__img"><img src="/-/media/Main-slider/VISA-EN.ashx"/></div>
        </a>
        """
        assert labels(extract_links(html, HOME)) == ["Banque Misr visa business cards"]

    def test_icon_link_falls_back_to_aria_label(self):
        html = '<a href="/Home/Pages/Branches" aria-label="Find a branch"><i class="pin"></i></a>'
        assert labels(extract_links(html, HOME)) == ["Find a branch"]

    def test_label_is_never_empty(self):
        # An unlabelled link is invisible to the LLM picking the next hop.
        html = """
        <a href="/Home/Pages/Credit%20Cards%20List"></a>
        <a href="/en/Digital-Services/apple-pay"><span></span></a>
        <a href="/"></a>
        """
        for link in extract_links(html, HOME):
            assert link["label"].strip()

    def test_slug_label_decodes_and_titlecases_only_when_needed(self):
        assert slug_label("https://www.banquemisr.com/Pages/Credit%20Cards%20List") == (
            "Credit Cards List"
        )
        assert slug_label("https://www.banquemisr.com/en/Digital-Services/apple-pay") == "Apple Pay"
        # Existing capitalisation is authoritative; .title() would give "Smes".
        assert slug_label("https://www.banquemisr.com/Home/SMEs") == "SMEs"

    def test_slug_label_skips_structural_segments(self):
        assert slug_label("https://www.banquemisr.com/Home/Pages/") == ""
        assert slug_label("https://www.banquemisr.com/Home/Pages/Fees/default.aspx") == "Fees"


class TestRegionTagging:
    def test_regions_are_detected(self):
        html = """
        <header class="site-header"><b-dropdown-item href="/Home/Pages/Cards">Cards</b-dropdown-item></header>
        <main><a href="/Home/Pages/Loans">Loans</a></main>
        <footer class="site-footer"><a href="/Home/Pages/Fees">Fees</a></footer>
        """
        links = extract_links(html, HOME)
        assert [link["source"] for link in links] == ["nav", "body", "footer"]

    def test_nav_inside_footer_is_reported_as_footer(self):
        html = '<footer><nav><a href="/Home/Pages/Fees">Fees</a></nav></footer>'
        assert extract_links(html, HOME)[0]["source"] == "footer"

    def test_class_names_are_used_when_semantic_tags_are_absent(self):
        html = '<div class="navbar-main"><a href="/Home/Pages/Cards">Cards</a></div>'
        assert extract_links(html, HOME)[0]["source"] == "nav"

    def test_unknown_region_defaults_to_body_and_never_drops(self):
        html = '<div class="whatever"><a href="/Home/Pages/Fees">Fees</a></div>'
        links = extract_links(html, HOME)
        assert len(links) == 1
        assert links[0]["source"] == "body"

    def test_footer_links_are_kept(self):
        # The fees hub is reachable only from the footer; filtering footers as
        # boilerplate makes a whole class of tasks unanswerable.
        html = '<footer><a href="/Home/Pages/Fees">Fees and Commissions</a></footer>'
        assert len(extract_links(html, HOME)) == 1


class TestOrderAndDedupe:
    def test_document_order_is_preserved(self):
        html = """
        <a href="/Home/Pages/A">A</a>
        <a href="/Home/Pages/B">B</a>
        <a href="/Home/Pages/C">C</a>
        """
        assert labels(extract_links(html, HOME)) == ["A", "B", "C"]

    def test_token_variants_collapse_to_one_entry(self):
        html = """
        <a href="/Home/Pages/Fees?csrt=aaa">Fees</a>
        <a href="/Home/Pages/Fees?csrt=bbb">Fees</a>
        <a href="/home/pages/fees">Fees</a>
        """
        assert len(extract_links(html, HOME)) == 1

    def test_duplicate_keeps_first_position_but_takes_the_better_label(self):
        html = """
        <nav><a href="/Home/Pages/Credit%20Cards%20List"></a></nav>
        <main><a href="/home/pages/credit-cards-list">Compare all credit cards</a></main>
        <a href="/Home/Pages/Other">Other</a>
        """
        links = extract_links(html, HOME)
        assert len(links) == 2
        assert links[0]["label"] == "Compare all credit cards"  # upgraded from the slug
        assert links[0]["source"] == "nav"  # first occurrence keeps the position

    def test_every_link_carries_its_canonical_key(self):
        html = '<a href="/Home/SMEs/Retail%20Banking">Retail</a>'
        link = extract_links(html, HOME)[0]
        assert link["key"] == canonical_key(link["url"])
        assert link["key"] == "banquemisr.com/home/smes/retail banking"


class TestPdfHints:
    def test_pdf_extension_is_a_hint(self):
        assert is_pdf_hint("https://www.banquemisr.com/Home/Pages/tariff.pdf") is True

    def test_sitecore_media_without_an_image_extension_is_a_hint(self):
        # PDFs are served from /-/media/... with a .ashx extension.
        assert is_pdf_hint("https://www.banquemisr.com/-/media/Fees/tariff-2024.ashx") is True

    def test_sitecore_media_images_are_not_pdf_hints(self):
        # The same prefix serves slider graphics; treating it as proof would
        # mark every homepage image as a downloadable tariff.
        assert is_pdf_hint("https://www.banquemisr.com/-/media/Main-slider/VISA.png") is False

    def test_ordinary_page_is_not_a_pdf_hint(self):
        assert is_pdf_hint("https://www.banquemisr.com/Home/Pages/Fees") is False

    def test_hint_is_surfaced_on_extracted_links(self):
        html = '<a href="/-/media/Fees/tariff.ashx">Download tariff</a>'
        assert extract_links(html, HOME)[0]["is_pdf"] is True


class TestNoiseSuppression:
    def test_links_whose_target_is_also_an_image_on_the_page_are_dropped(self):
        html = """
        <img src="/-/media/Main-slider/VISA-EN.ashx"/>
        <a href="/-/media/Main-slider/VISA-EN.ashx">View</a>
        <a href="/Home/Pages/Fees">Fees</a>
        """
        assert labels(extract_links(html, HOME)) == ["Fees"]

    def test_rejected_links_never_appear(self):
        html = """
        <a href="mailto:info@banquemisr.com">Email</a>
        <a href="https://www.facebook.com/BanqueMisr">Facebook</a>
        <a href="/ar-eg/Home/Pages/Default">Arabic</a>
        <a href="/Home/Pages/Fees">Fees</a>
        """
        assert labels(extract_links(html, HOME)) == ["Fees"]


class TestAgainstFixtures:
    def test_homepage_fixture_exercises_every_quirk(self, fixture_html):
        links = extract_links(fixture_html("homepage.html"), HOME)
        found = {link["source"] for link in links}
        assert found == {"nav", "body", "footer"}
        assert by_key(links, "https://www.banquemisr.com/Home/Pages/Fees")["source"] == "footer"
        assert all(link["label"].strip() for link in links)
        assert not any("facebook" in link["url"] for link in links)
        assert not any("ar-eg" in link["url"].lower() for link in links)


class TestRepeatedLabelDisambiguation:
    """Same text, different destinations -- the selector needs a tiebreak.

    Sitecore builds category tiles from one template, so every tile on a hub
    reads "View more details". A label shared by eleven links identifies none
    of them.
    """

    HTML = """
    <html><body><main>
      <a href="/things/red">More details</a>
      <a href="/things/blue">More details</a>
      <a href="/things/green">Green things</a>
      <a href="/help">Help</a>
    </main></body></html>
    """

    def links(self):
        return {
            l["url"].rsplit("/", 1)[-1]: l["label"]
            for l in extract_links(self.HTML, "https://www.banquemisr.com/")
        }

    def test_repeated_labels_gain_what_the_url_says(self):
        labels = self.links()
        assert labels["red"] == "More details - Red"
        assert labels["blue"] == "More details - Blue"

    def test_a_label_that_appears_once_is_untouched(self):
        labels = self.links()
        assert labels["green"] == "Green things"
        assert labels["help"] == "Help"

    def test_nothing_is_appended_when_the_slug_only_repeats_the_label(self):
        html = """
        <html><body><main>
          <a href="/news">News</a>
          <a href="/press/news">News</a>
        </main></body></html>
        """
        assert [
            l["label"] for l in extract_links(html, "https://www.banquemisr.com/")
        ] == ["News", "News"]

    def test_the_flag_restores_the_raw_anchor_text(self, monkeypatch):
        monkeypatch.setattr(config, "DISAMBIGUATE_REPEATED_LABELS", False)
        assert self.links()["red"] == "More details"
