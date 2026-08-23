# Person A ↔ Person B Integration Contract

## Status

This is the integration contract for two **separate repositories**.

- Person A owns navigation/fetching.
- Person B owns planning/extraction/validation/reasoning/verification.
- Final integration occurs in Person A's repository after Person B is complete.

---

# 1. Design Principle

Person B must never import or depend on Person A internal modules.

Integration happens by passing plain values/serializable records through the public Person B API.

---

# 2. Mandatory Compatibility Function

Person A explicitly needs:

```python
validate(sub_goal, page_content)
```

with a result containing:

```python
{
    "resolved": bool,
    "extracted": {...},
    "reason": str,
}
```

This two-argument form is **mandatory**.

Person B may accept optional keyword context, but must not break this call.

---

# 3. Recommended Enriched Validation Input

For production-quality source attribution and PDF handling, Person A should pass additional metadata when available.

Conceptual signature:

```python
validate(
    sub_goal,
    page_content,
    *,
    source_url=None,
    content_type=None,
    status_code=None,
    pdf_bytes=None,
    pdf_path=None,
    pdf_tables=None,
    metadata=None,
)
```

## Required minimum

- `sub_goal`
- `page_content`

## Strongly recommended

- `source_url`
- `content_type`

## PDF-specific optional values

At least one of:
- `pdf_bytes`
- `pdf_path`
- `pdf_tables`

when table structure is necessary.

---

# 4. Why PDF Material Matters

The supplied cross-card fee PDF loses reliable card-column mapping if treated only as flat text.

Person B therefore needs table-aware input for such documents.

Person A should not assume:

```text
flattened PDF text == sufficient structured evidence
```

If only flattened text is available and the mapping is ambiguous, Person B must report insufficient evidence rather than guess.

---

# 5. Link Records From Person A

Latest known link record:

```python
{
    "label": str,
    "url": str,
    "source": "nav" | "body" | "footer",
    "is_pdf": bool,
    "key": str,
}
```

Rules:

- `key` belongs to Person A's visited-set; Person B should not depend on its format.
- `is_pdf` is a hint only.
- actual fetched `content_type` wins.

---

# 6. Validation Output

Minimum:

```python
{
    "resolved": bool,
    "extracted": dict,
    "reason": str,
}
```

Recommended richer output:

```python
{
    "resolved": bool,
    "status": "resolved" | "partial" | "unresolved" | "unreadable",
    "extracted": dict,
    "evidence": list,
    "missing_fields": list,
    "reason": str,
    "source_url": str | None,
}
```

Person A should use `resolved` as the navigation stop/continue signal.

---

# 7. Partial Results

A page may provide valid evidence without fully satisfying the sub-goal.

Person B may return:

```python
{
    "resolved": False,
    "status": "partial",
    "extracted": {
        "benefits": [...]
    },
    "missing_fields": ["fees"],
    "reason": "Benefits found; fee evidence still missing."
}
```

Person A should preserve the extracted partial evidence while continuing navigation.

---

# 8. Planning Contract

Person B should expose a public planning call that returns a serializable plan.

Conceptually:

```python
plan = plan_task(user_task)
```

A plan should identify:
- the normalized goal;
- sub-goals;
- IDs;
- statuses;
- dependencies if any.

Person A should not manually mutate deep planner internals.

Person B should expose helper functions/methods for:
- next pending sub-goal;
- applying validation result;
- dynamic plan expansion;
- marking not available.

---

# 9. Navigation Contract

Person A owns:

```python
navigate(sub_goal)
```

or an equivalent candidate-page iterator.

Person B does **not** dictate Person A's internal navigation implementation.

The only expected behavior is:

- Person A obtains a page;
- Person A passes it to Person B validation;
- Person A continues if unresolved;
- Person A stops for that sub-goal when resolved or limits are exhausted.

---

# 10. Source/Evidence Contract

For every source used in final factual output, Person B should be able to retain:

- source URL;
- evidence text or table cell;
- extracted field/value;
- optional page/table location metadata.

Person A should preserve real visited URLs.

---

# 11. Final Verification Contract

Before final output, Person B should receive enough run context to determine whether every factual claim is supported by a visited source.

Conceptually:

```python
result = validate_answer(
    claims=draft_claims,
    visited_pages=visited_pages,
)
```

A failed claim must not survive unchanged into the final answer.

---

# 12. Unreadable PDF Contract

If a PDF returns no text/tables:

```python
{
    "resolved": False,
    "status": "unreadable",
    "extracted": {},
    "reason": "PDF returned no extractable text/tables."
}
```

Normal unreadable content is a structured outcome, not a crash.

---

# 13. Source URL Missing

The minimum validation call may omit a URL.

In that case:
- extraction/validation may proceed;
- evidence should indicate URL unavailable;
- final sourced output should request/rely on the enriched integration path for full source coverage.

Production integration should pass URLs.

---

# 14. No Hidden Dependency Contract

Person B must not assume:
- Person A folder names;
- Person A classes;
- Person A logger implementation;
- Person A HTTP client;
- Person A navigation state type;
- Person A runtime model SDK.

Person A must not assume:
- Person B internal module names beyond the documented public facade.

---

# 15. Integration Compatibility Tests

Person B tests must include:

```python
def test_validate_minimum_contract():
    result = validate("some sub-goal", "some page content")
    assert set(["resolved", "extracted", "reason"]) <= result.keys()
```

and enriched-context tests.

---

# 16. Handoff Deliverables

Person B hands Person A:

- `src/person_b/`
- dependency configuration
- `README.md`
- this contract
- tests/integration smoke example if desired
- release/version identifier

Person A should be able to clone the repository and integrate without requesting undocumented internal details.
