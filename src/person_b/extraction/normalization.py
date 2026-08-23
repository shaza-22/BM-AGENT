"""Normalization helpers for extracted text, currency, numbers, and entity names."""

import re
from typing import Any, Optional


def normalize_whitespace(text: str) -> str:
    """Normalize multiple whitespace characters and clean non-breaking spaces."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_field_value(value: Any) -> Any:
    """Normalize arbitrary extracted field value."""
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, (list, tuple)):
        return [normalize_field_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_field_value(v) for k, v in value.items()}
    return value


def normalize_entity_name(name: str) -> str:
    """Standardize product/entity names (e.g. 'Gold Credit Cards' -> 'Gold Credit Card')."""
    cleaned = normalize_whitespace(name)
    # Strip leading bullets or symbols using unicode escapes
    cleaned = cleaned.lstrip("\u2022\u2023\u25e6\u2043\u2219\u00b7\u25aa\u25ab-* \t\n\r")
    # Normalize common plural forms in card names
    cleaned = re.sub(r"(?i)\bcredit cards\b", "Credit Card", cleaned)
    cleaned = re.sub(r"(?i)\bmastercard\b", "MasterCard", cleaned)
    return cleaned.strip()


def extract_currency_value(text: str) -> Optional[dict]:
    """Extract numeric amount and currency from text if present."""
    if not text:
        return None
    # Matches patterns like 'EGP 250', '250 EGP', '250EGP', '50 USD', '30000 EGP Daily'
    pattern = r"(?:(?P<curr1>EGP|USD|EUR|GBP|LE)\s*)?(?P<num>\d+(?:,\d+)*(?:\.\d+)?)\s*(?:(?P<curr2>EGP|USD|EUR|GBP|LE))?"
    m = re.search(pattern, text, re.IGNORECASE)
    if m and (m.group("curr1") or m.group("curr2")):
        curr = (m.group("curr1") or m.group("curr2")).upper()
        if curr == "LE":
            curr = "EGP"
        raw_num = m.group("num").replace(",", "")
        try:
            val = float(raw_num) if "." in raw_num else int(raw_num)
            return {"amount": val, "currency": curr, "raw": text.strip()}
        except ValueError:
            pass
    return None
