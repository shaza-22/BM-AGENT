# TASK 02 — Planning, Dynamic Expansion, and Extraction

## Anti-Gravity Execution Profile

- Model: Gemini 3.7 Flash
- Reasoning: High / maximum available
- Scope: Person B planning/extraction only

---

# Objective

Implement Person B's complete:

- task understanding;
- initial planning;
- dynamic plan expansion;
- cleaned-text preprocessing;
- pipe-table extraction;
- PDF table extraction;
- generic structured extraction;
- evidence lineage.

At the end of this task, Person B must be able to turn a task into sub-goals and extract reliable structured evidence from supplied real fixtures.

Validation/reasoning/self-check are completed in Task 03.

---

# Mandatory Preflight

Read:

1. `AGENT_START_HERE.md`
2. `docs/MASTER_PLAN.md`
3. `docs/contracts/PERSON_A_PERSON_B_CONTRACT.md`
4. `README.md`
5. `docs/tasks/TASK_01_FOUNDATION_AND_CONTRACTS.md`
6. this task file

Inspect all code implemented in Task 01 before modifying it.

Run the Task 01 tests before starting.

---

# Part A — Task Understanding

Implement task normalization that preserves intent.

Input:

```python
user_task: str
```

Output concept:

```python
Goal(
    original_task=...,
    statement=...
)
```

Requirements:

- do not add unstated facts;
- preserve requested entities/constraints;
- distinguish lookup vs comparison vs recommendation vs process/multi-step research when useful;
- keep the design generic.

If runtime reasoning backend is used, validate its structured output before accepting it.

---

# Part B — Initial Planning

Implement:

```python
plan_task(user_task)
```

The initial plan should create the **minimum useful** sub-goals.

Example:

```text
User:
"Find all Banque Misr credit cards, compare their fees and benefits."

Good initial plan:
1. Find the list of Banque Misr credit cards.

Not ideal:
hardcode 12 specific cards before the site has revealed them.
```

Sub-goals need stable IDs and state.

---

# Part C — Dynamic Plan Expansion

Implement an expansion mechanism such as:

```python
expand_plan(plan, validation_or_extraction_result)
```

Use new evidence to add new research work.

Example:

```text
Validated credit-card-list page
-> extracted card names
-> add sub-goal(s) to obtain requested dimensions for the discovered cards
```

Requirements:

- deduplicate semantically/equivalently repeated sub-goals;
- do not re-add already resolved work;
- maintain parent/created-from relationship where useful;
- support bounded growth;
- preserve deterministic order where possible.

Do not hardcode Banque Misr card names.

---

# Part D — Cleaned Text Processing

Person A's `.txt` output is the primary HTML-page input.

Implement preprocessing that:

- normalizes line endings;
- preserves line order;
- preserves useful headings;
- recognizes likely repeated nav/footer boilerplate;
- can create a content-focused view for reasoning;
- keeps raw content for evidence snippets.

Important:

> Do not destructively remove content if it could contain evidence.

A practical design can retain both:
- `raw_text`
- `focused_text`

---

# Part E — Pipe-delimited Table Parser

Implement deterministic parsing for patterns such as:

```text
limits | Details
Limit of Online Purchasing Inside Egypt | ...
Local cash withdrawals (ATM) | EGP 30000 Daily
```

and:

```text
Fees and charges | Details
Issuance | EGP 250
Renewal | EGP 250
```

Handle:

- 2+ columns;
- optional Markdown separator row;
- spaces around `|`;
- multiple tables in one page;
- table title/header immediately before table;
- blank lines/prose around tables;
- uneven rows;
- duplicate row names;
- multiline artifacts where possible.

Represent tables in a reusable structured shape.

---

# Part F — PDF Table Extraction

Use the real fee PDF fixture.

Requirements:

1. use a table-aware method such as `pdfplumber.extract_tables()`;
2. iterate over every PDF page;
3. iterate over every table on each page;
4. normalize multiline cells;
5. normalize `None`;
6. retain page/table index for evidence;
7. detect likely header rows;
8. do not rely on flat `extract_text()` for cross-column mapping;
9. tolerate small one-column artifacts produced by rotated/split labels;
10. keep enough raw table shape to avoid incorrect column shifts.

The supplied fee PDF contains many card columns; table integrity is critical.

---

# Part G — Image-only PDF

Use the supplied ATM activation guide.

Expected extraction result:

- no fabricated text;
- status indicating unreadable/no extractable content;
- warning/reason;
- no crash.

Do not introduce OCR as a hidden fallback unless explicitly required by a later project decision.

The handoff treats this as a legitimate unavailable-information case.

---

# Part H — Generic `extract_content`

Implement a public/internal extraction function conceptually like:

```python
extract_content(
    page_content,
    target_fields,
    *,
    source_url=None,
    content_type=None,
    pdf_bytes=None,
    pdf_path=None,
    pdf_tables=None,
    ...
)
```

Requirements:

- generic target fields;
- deterministic table extraction first where appropriate;
- optional reasoning backend for interpretation/normalization;
- evidence attached to values;
- structured missing-fields list;
- parser warnings;
- no invented values.

---

# Part I — Evidence Lineage

Each useful extracted field must know where it came from.

At minimum, preserve:

- target field;
- extracted value;
- source URL if known;
- evidence snippet/cell;
- evidence type;
- table/page location where available.

For PDF:

```text
page number
table index
row/column context
```

For text tables:

```text
table heading if available
row/cell text
```

---

# Part J — Duplicate/Conflicting Evidence

The implementation must not silently overwrite when two sources contain the same field.

Example:

- Classic product page has an issuance fee;
- fee PDF also has an issuance fee.

Represent multiple evidence records or a conflict state.

If values agree:
- they may strengthen confidence;
- retain both sources if useful.

If they differ:
- preserve both;
- do not choose silently;
- let later reasoning/validation decide based on context/source recency/coverage if such metadata exists.

---

# Part K — Real Fixture Tests

Use the supplied cleaned fixtures to test:

## Cards category
Extract card-category content without confusing nav links.

## Credit Cards list
Extract individual card names.

Expected examples present in the fixture include:
- Classic Credit Card
- Gold Credit Cards
- Titanium Credit Card
- Visa Infinite
- Visa Signature
- World Credit Card
- etc.

Do not hardcode the expected list in production logic. It is acceptable to assert known fixture values in tests.

## Classic Credit Card
Extract:
- benefits;
- usage limits table;
- credit limit;
- fees/charges table;
- installment table.

## Accounts & Deposits
Prove body extraction works despite misleading first-line title.

## Consumer Loans
Extract multiple loan offerings and descriptive details.

## Fees hub
Extract attachment/document labels such as the cards fee/limits document.

## Fee PDF
Prove multi-card table mapping remains intact.

## Image-only PDF
Prove graceful unreadable result.

---

# Part L — Planner Tests

Add tests for:

1. direct lookup;
2. comparison;
3. recommendation;
4. multi-step research;
5. dynamic expansion after entity discovery;
6. no duplicate expansion;
7. no hardcoded card list.

Use a fake deterministic reasoning backend where needed.

---

# Non-goals

Do not implement:

- Person A navigation;
- live HTTP;
- final validation decision logic;
- final recommendation engine;
- final claim self-check.

---

# Quality Checks

Run:

```bash
python -m pytest -q
python -m compileall src
```

and configured lint checks.

No test may require the live Banque Misr website.

---

# Deliverable Report

Report:

1. planner behavior;
2. dynamic expansion behavior;
3. parsers implemented;
4. PDF approach;
5. evidence schema;
6. fixture tests added;
7. test results;
8. any unresolved PDF/fixture limitations.

---

# Definition of Done

- [ ] `plan_task()` works;
- [ ] dynamic expansion works;
- [ ] sub-goal dedupe works;
- [ ] cleaned text preprocessing works;
- [ ] pipe tables parse correctly;
- [ ] fee PDF tables parse without flattening columns;
- [ ] image-only PDF fails gracefully;
- [ ] `extract_content()` is generic;
- [ ] evidence lineage exists;
- [ ] conflicting evidence is not silently overwritten;
- [ ] fixture tests pass;
- [ ] no Person A code was implemented.
