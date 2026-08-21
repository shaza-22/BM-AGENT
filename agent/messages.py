"""
The agent's own sentences, in the language of the request.

What it does
    Holds every user-facing string the code writes itself -- as opposed to the
    ones the model writes -- and returns them in the run's language.

Inputs
    ``message(key, language, **values)``.

Outputs
    The formatted sentence. Falls back to English for an unknown language or an
    unknown key, never raising: a missing translation must not fail a run.

Why it is needed
    The model is instructed to explain its choices in the language it was asked
    in, so an Arabic run's reasoning comes back in Arabic. But the outcomes the
    code decides -- running out of links, hitting a cap, failing to parse a
    reply -- were written here, in English, and those are exactly the cases
    where a user most needs to understand what happened. An Arabic run that
    explains its successes in Arabic and its failures in English is worse than
    one that is consistent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MESSAGES: dict[str, dict[str, str]] = {
    "no_links_left": {
        "en": "No unvisited links remain to choose from.",
        "ar": "لم تعد هناك روابط جديدة يمكن اختيارها.",
    },
    "model_reported_arrived": {
        "en": "The model reported that the current page appears to be the destination.",
        "ar": "أفاد النموذج بأن الصفحة الحالية هي الوجهة المطلوبة على ما يبدو.",
    },
    "model_reported_none": {
        "en": "The model reported that no listed link leads toward the sub-goal.",
        "ar": "أفاد النموذج بعدم وجود رابط يقود إلى الهدف المطلوب.",
    },
    "parse_failed": {
        "en": "Could not parse the model's reply ({error}); treating as no candidate.",
        "ar": "تعذّر تحليل رد النموذج ({error})، وسيُعامل على أنه لا يوجد اختيار.",
    },
    "index_out_of_range": {
        "en": "The model chose link {index}, which was not on the list of {count} offered; "
              "treating as no candidate.",
        "ar": "اختار النموذج الرابط رقم {index} وهو ليس ضمن الروابط المعروضة "
              "وعددها {count}، وسيُعامل على أنه لا يوجد اختيار.",
    },
    "no_explanation": {
        "en": "Selected {label!r} but gave no explanation.",
        "ar": "تم اختيار {label!r} دون توضيح السبب.",
    },
    "below_confidence": {
        "en": "Chose {label!r} with confidence {confidence:.2f}, below the configured "
              "minimum of {minimum:.2f}.",
        "ar": "تم اختيار {label!r} بثقة {confidence:.2f}، وهي أقل من الحد الأدنى "
              "المحدد {minimum:.2f}.",
    },
    "cap_hops": {
        "en": "Reached the maximum of {cap} hops without resolving the sub-goal.",
        "ar": "تم بلوغ الحد الأقصى وهو {cap} خطوات دون الوصول إلى إجابة.",
    },
    "cap_pages": {
        "en": "Reached the maximum of {cap} pages without resolving the sub-goal.",
        "ar": "تم بلوغ الحد الأقصى وهو {cap} صفحات دون الوصول إلى إجابة.",
    },
    "blocked": {
        "en": "The site returned an access-denied page; stopping to avoid a ban.",
        "ar": "أعاد الموقع صفحة رفض وصول، وتم إيقاف البحث تجنبًا للحظر.",
    },
    "llm_unreachable": {
        "en": "The language model could not be reached: {error}",
        "ar": "تعذّر الوصول إلى النموذج اللغوي: {error}",
    },
    "seed_step": {
        "en": "Starting page for every run (the seed URL).",
        "ar": "صفحة البداية لكل عملية بحث (الرابط الأساسي).",
    },
    "no_validator": {
        "en": "no validator installed (stub); navigation will run to its cap",
        "ar": "لا يوجد مدقق مثبت (نسخة مبدئية)؛ سيستمر البحث حتى الحد الأقصى",
    },
}

DEFAULT_LANGUAGE = "en"


def message(key: str, language: str | None = None, **values: Any) -> str:
    """The sentence for *key* in *language*, formatted with *values*."""
    variants = MESSAGES.get(key)
    if variants is None:  # pragma: no cover - a typo in a call site
        logger.warning("no message registered for %r", key)
        return key
    template = variants.get((language or DEFAULT_LANGUAGE), variants[DEFAULT_LANGUAGE])
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        logger.warning("could not format message %r for %r", key, language)
        return variants[DEFAULT_LANGUAGE]
