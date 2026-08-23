"""Verify offline fixture loader against real supplied Banque Misr fixtures."""

from pathlib import Path
import pytest
from person_b.adapters.fixture_loader import FixtureLoader
from person_b.errors import FixtureError


def test_fixture_loader_manifest(fixture_loader: FixtureLoader):
    assert fixture_loader.exists(), "Fixtures directory must exist"
    manifest = fixture_loader.get_manifest()

    assert "pages" in manifest
    assert isinstance(manifest["pages"], list)
    assert len(manifest["pages"]) >= 7


def test_fixture_loader_load_cleaned_text(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_text("home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards.txt")
    assert ctx.content_type == "text"
    assert len(ctx.content) > 100
    assert "Classic" in ctx.content


def test_fixture_loader_load_html(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_html("home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards.html")
    assert ctx.content_type == "html"
    assert "<html" in ctx.content.lower() or "<!doctype html>" in ctx.content.lower()


def test_fixture_loader_load_pdf(fixture_loader: FixtureLoader):
    ctx = fixture_loader.load_pdf("usage-limits-and-fees-en.pdf")
    assert ctx.content_type == "pdf"
    assert ctx.pdf_bytes is not None
    assert len(ctx.pdf_bytes) > 1000
    assert ctx.pdf_path is not None


def test_fixture_loader_load_page_by_url(fixture_loader: FixtureLoader):
    url = "https://www.banquemisr.com/Home/SMEs/Retail%20Banking/Pages/Cards/Credit%20Cards%20Pages/Classic%20Credit%20Cards"
    ctx = fixture_loader.load_page_by_url(url)
    assert ctx.source_url == url
    assert len(ctx.content) > 0


def test_fixture_loader_missing_file_raises_error(fixture_loader: FixtureLoader):
    with pytest.raises(FixtureError):
        fixture_loader.load_text("nonexistent_fixture_file.txt")
