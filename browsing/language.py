"""
Script-based language detection.

What it does
    Decides whether a piece of text is Arabic or Latin-script, by counting
    letters. No model call, no dependency, no network.

Inputs
    ``detect_language(text, default=...)`` -- any string: a user's task, a link
    label, a page title.

Outputs
    A language code (``"ar"`` / ``"en"``), or ``default`` when the text carries
    no letters at all to judge by.

Why it is needed
    The agent has to know which language a task is in before it fetches
    anything, so it can seed the right side of a bilingual site and answer in
    the language it was asked in. Spending a model request on something Unicode
    already tells us would be waste -- the free tier allows about twenty
    requests a day, and the session resolver already claims one per follow-up.

Why a ratio rather than a raw count
    Real requests mix scripts. "BM Wallet ازاى" is an Arabic question with an
    English product name in it, and a raw count would call it English. The
    threshold is deliberately below half (see
    :data:`browsing.config.ARABIC_DETECTION_THRESHOLD`): Latin tokens inside
    Arabic questions are common -- brand names, "BM", "Visa" -- while Arabic
    tokens inside English questions are rare. The asymmetry in the data is why
    the threshold is asymmetric.
"""

from __future__ import annotations

import unicodedata

from browsing import config

# Arabic proper, plus the Supplement and Extended-A blocks and both
# Presentation Forms ranges, which Sitecore and copy-paste both produce.
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

# Both digit sets are excluded from the ratio: a number says nothing about the
# language of the sentence around it.
_ARABIC_DIGITS = ((0x0660, 0x0669), (0x06F0, 0x06F9))


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= code <= high for low, high in ranges)


def is_arabic_char(char: str) -> bool:
    code = ord(char)
    return _in_ranges(code, _ARABIC_RANGES) and not _in_ranges(code, _ARABIC_DIGITS)


def is_latin_letter(char: str) -> bool:
    if not char.isalpha():
        return False
    try:
        return "LATIN" in unicodedata.name(char)
    except ValueError:  # unnamed character
        return False


def script_counts(text: str) -> tuple[int, int]:
    """``(arabic_letters, latin_letters)`` in *text*. Digits and punctuation ignored."""
    arabic = latin = 0
    for char in text or "":
        if is_arabic_char(char):
            arabic += 1
        elif is_latin_letter(char):
            latin += 1
    return arabic, latin


def arabic_ratio(text: str) -> float:
    """Share of the letters that are Arabic. ``0.0`` when there are no letters."""
    arabic, latin = script_counts(text)
    total = arabic + latin
    return arabic / total if total else 0.0


def detect_language(text: str, *, default: str | None = None) -> str | None:
    """Language of *text*, or *default* when there is nothing to judge by.

    *default* falls back to :data:`browsing.config.LANGUAGE` -- read at call
    time, so a per-run override or a test monkeypatch takes effect.
    """
    fallback = default if default is not None else config.LANGUAGE
    arabic, latin = script_counts(text)
    if arabic + latin == 0:
        return fallback
    if arabic / (arabic + latin) >= config.ARABIC_DETECTION_THRESHOLD:
        return "ar"
    return "en"
