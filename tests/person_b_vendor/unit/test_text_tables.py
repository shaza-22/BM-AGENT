"""Unit tests for pipe-delimited table parser."""

from person_b.extraction.text_tables import parse_text_tables
from person_b.models import ExtractionLocationType


def test_parse_simple_two_column_table():
    text = """
Fees and charges | Details
Issuance | EGP 250
Renewal | EGP 250
Interest rate | 4% monthly
"""
    tables = parse_text_tables(text, source_url="https://banquemisr.com/test")
    assert len(tables) == 1
    t = tables[0]
    assert t["headers"] == ["Fees and charges", "Details"]
    assert len(t["rows"]) == 3
    assert t["records"][0]["Fees and charges"] == "Issuance"
    assert t["records"][0]["Details"] == "EGP 250"
    assert len(t["evidence"]) == 6
    assert t["evidence"][0].location.type == ExtractionLocationType.TEXT_TABLE


def test_parse_table_with_markdown_separator():
    text = """
Tenor | Interest Rate
--- | ---
3 | 2.81%
6 | 2.77%
12 | 2.73%
"""
    tables = parse_text_tables(text)
    assert len(tables) == 1
    t = tables[0]
    assert t["headers"] == ["Tenor", "Interest Rate"]
    assert len(t["rows"]) == 3
    assert t["records"][0] == {"Tenor": "3", "Interest Rate": "2.81%"}


def test_parse_table_with_preceding_heading():
    text = """
Usage limits
limits | Details
Local cash withdrawals (ATM) | EGP 30000 Daily
International purchase transactions | 50,000EGP Monthly
"""
    tables = parse_text_tables(text)
    assert len(tables) == 1
    t = tables[0]
    assert t["table_name"] == "Usage limits"
    assert len(t["rows"]) == 2


def test_parse_multiple_tables_in_single_text():
    text = """
Table One
Col A | Col B
val1 | val2

Some prose paragraph in between.

Table Two
Col X | Col Y
valX | valY
"""
    tables = parse_text_tables(text)
    assert len(tables) == 2
    assert tables[0]["table_name"] == "Table One"
    assert tables[1]["table_name"] == "Table Two"
