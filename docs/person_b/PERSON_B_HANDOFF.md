# Person B Release & Handoff Document
## Banque Misr Agentic Research Assistant ? Person B (Intelligence Layer)

- **Version:** `0.1.0`
- **Release Status:** `READY FOR PERSON A INTEGRATION`
- **Python Compatibility:** Python >= 3.10 (tested on Python 3.14.6)
- **Zero Live Network Calls Required in Tests:** Confirmed

---

## 1. What Person B Delivers

Person B is the complete, standalone intelligence layer for the Banque Misr Agentic Research Assistant. It provides:

1. **Task Understanding & Planning (`person_b.plan_task`)**: Converts natural language requests into normalized goals and minimal initial sub-goals (e.g. discovery sub-goal for broad comparisons, target sub-goal for direct lookups).
2. **Dynamic Plan Expansion (`person_b.expand_plan`)**: Discovers products/entities from validated pages and creates targeted child sub-goals with deterministic deduplication and expansion bounds.
3. **Deterministic Pipe-Table Parsing (`person_b.extraction.text_tables`)**: Extracts Markdown-style and pipe-delimited tables (`Col A | Col B`) with headings, row records, and cell-level provenance.
4. **Table-Aware PDF Extraction (`person_b.extraction.pdf_tables`)**: Parses multi-page cross-card fee PDFs via `pdfplumber`, keeping multi-column card headers aligned without loss.
5. **Robust False-Positive & Misleading-Title Defenses**: Uses body headings, focused text, and breadcrumbs; ignores nav/footer boilerplate and misleading line-1 titles.
6. **Sub-Goal Validation Engine (`person_b.validate`)**: Implements both the mandatory 2-positional-argument contract and enriched keyword arguments, supporting `resolved`, `partial`, `unresolved`, and `unreadable` states.
7. **Structured Comparison (`person_b.compare_items`)**: Compares products across arbitrary dimensions, preserving missing values (`Not Available`) and evidence provenance without fabricating data.
8. **Evidence-Grounded Recommendation (`person_b.recommend`)**: Matches user criteria against validated evidence, separating factual premises from recommendation judgment while disclosing tradeoffs.
9. **Synthesis & Explicit Claim Generation (`person_b.synthesize`)**: Aggregates multi-source evidence into draft answers and converts every factual assertion into an explicit `Claim` object.
10. **Strict Anti-Hallucination Verification (`person_b.validate_answer`, `person_b.finalize`)**: Verifies every claim against visited source URLs during that run, rejects overbroad superlatives, and removes unverified statements.
11. **Structured JSONL Logging (`person_b.PersonBLogger`)**: Structured events for task understanding, plan creation, extraction, validation, synthesis, and verification.
12. **Fixture-Based Benchmark Evaluation (`person_b.PersonBEvaluator`)**: 8 benchmark test categories with quantitative metrics (`validation_accuracy`, `false_positive_rate`, `field_extraction_correctness`, `claim_support_rate`, `source_attribution_rate`, `dynamic_expansion_success`).

---

## 2. What Person A Needs to Copy / Install

### Production Package
To integrate Person B into Person A's repository, Person A only needs:
- `src/person_b/` directory (or install the repository via `pip install -e .`)
- Dependencies from `requirements.txt` (`python-dotenv>=1.0.0`, `pdfplumber>=0.10.0`)

### Optional Development & Testing Files
- `examples/person_a_integration_smoke.py` (Runnable smoke test)
- `fixtures/live/` (Real Banque Misr HTML/text/PDF fixtures for offline testing)
- `tests/` (Full offline test suite)

---

## 3. Public API Summary

```python
from person_b import (
    # Planning
    plan_task,
    next_pending_sub_goal,
    apply_validation,
    mark_not_available,
    expand_plan,
    # Extraction
    extract_content,
    # Validation
    validate,
    # Reasoning
    compare_items,
    recommend,
    synthesize,
    # Verification
    attribute_sources,
    validate_answer,
    finalize,
    # Adapters & Tools
    FixtureLoader,
    PersonBLogger,
    PersonBEvaluator,
    PersonBConfig,
)
```

---

## 4. Mandatory Compatibility Contract

The minimum Person A contract is strictly preserved:

```python
result = validate(sub_goal, page_content)

assert "resolved" in result
assert "extracted" in result
assert "reason" in result
```

For production source attribution and PDF handling, pass enriched arguments:

```python
result = validate(
    sub_goal,
    page_content,
    source_url=page.url,
    content_type=page.content_type,
    status_code=page.status_code,
    pdf_bytes=page.pdf_bytes,          # optional
    pdf_tables=page.pdf_tables,        # optional
)
```

---

## 5. Test & Evaluation Verification Summary

- **Total Test Count:** 63 tests in pytest
- **Pass Rate:** 100% (63 passed, 0 failed)
- **Evaluation Suite Score:** 100% (8/8 benchmark categories passed)
- **Live Network Requests:** 0 (100% offline against local fixtures)

---

## 6. Integration Starting Point

Person A can immediately verify the integrated flow offline by running:

```bash
python examples/person_a_integration_smoke.py
```
