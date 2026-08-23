# Supplied Source Notes

This file records the inputs used to build the Person B plan.

## Project documents

- `BM_Agentic_Research_Assistant_Build_Plan.docx`
- `handoff_to_person_b.md.docx`

## Real fixture package

The supplied `live.zip` contains cleaned text/raw HTML fixtures for:

- Banque Misr homepage
- Cards category
- Credit Cards list
- Classic Credit Card detail
- Accounts & Deposits
- Consumer Loans
- Fees hub

It also contains:
- a cross-card usage/limits/fees PDF;
- an image-only ATM activation guide PDF;
- `manifest.json`.

## Design facts carried into implementation

1. Cleaned `.txt` is the primary HTML-page input shape Person A expects to hand Person B at runtime.
2. Cleaned text can preserve tables with `|` separators.
3. Repeated navigation/footer boilerplate appears before/after body content.
4. The fee PDF needs table-aware extraction to preserve card-column mapping.
5. The image-only PDF must fail gracefully without hallucination.
6. Person A's latest link shape includes `label`, `url`, `source`, `is_pdf`, and `key`.
7. Actual fetched `content_type` is more trustworthy than the link `is_pdf` hint.
8. Person A needs `validate(sub_goal, page_content)` returning at least `resolved`, `extracted`, and `reason`.
9. Person A owns `navigate()` and the final shared navigation loop.
10. The supplied Accounts & Deposits fixture has a misleading first-line page title, so validation cannot trust title alone.
11. The supplied Classic Credit Card cleaned fixture contains a fees table even though the handoff emphasizes the central fee PDF, so extraction must be source/evidence-driven rather than hardcoded to one fee location.
