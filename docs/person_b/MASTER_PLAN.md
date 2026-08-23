# Master Plan — Person B
## Banque Misr Agentic Research Assistant

**Repository role:** standalone Person B implementation repository  
**Implementation agent:** Anti-Gravity CLI  
**Implementation model:** Gemini 3.7 Flash  
**Reasoning level for implementation:** High / maximum available

---

# 1. Mission

Complete **all Person B responsibilities** for the Banque Misr Agentic Research Assistant in this repository, independently from Person A's codebase.

The finished repository must be:

- complete enough for Person B's owned functionality;
- tested against real Banque Misr fixtures;
- generic across lookup, comparison, recommendation, and multi-step research;
- evidence-first and resistant to hallucination;
- portable into Person A's repository;
- documented so Person A can integrate it with minimal back-and-forth.

The end-state is:

> Person B is finished, tested, documented, and integration-ready.  
> Person A can clone this repository, take the Person B package, and connect it to Person A's navigation loop.

---

# 2. Product Context

The complete application is a live web-browsing agent that:

1. starts only from the Banque Misr homepage;
2. understands the user's request;
3. plans sub-goals;
4. navigates the live website dynamically;
5. extracts actual content from reached pages;
6. validates whether each sub-goal is truly answered;
7. reasons/synthesizes across gathered evidence;
8. self-checks every factual claim against sources visited during that run;
9. returns a final answer with exact source URLs and explicit gaps.

There is no pre-built search index for task answers and no fixed per-task navigation script.

The same pipeline must support:

- direct lookup;
- product discovery;
- comparison;
- recommendation;
- multi-hop research;
- deliberate trap questions that the website cannot answer.

---

# 3. Ownership Split

## 3.1 Person A — outside this repository

Person A owns the live-navigation side:

- `fetch_page(url)`
- live requests
- text cleaning performed by the fetcher
- `extract_links(page_content)`
- link selection
- `navigate(sub_goal)`
- canonical visited-set key
- navigation retries
- hop/page limits
- WAF-safe pacing
- live-site request policy
- final combined agent loop inside Person A's repository
- unrelated backend/frontend work

Person B must **not** recreate these responsibilities.

## 3.2 Person B — this repository

Person B owns:

### Task understanding
Turn the raw user request into a precise goal.

### Planning
Break the goal into sub-goals and maintain their state.

### Dynamic plan expansion
When evidence reveals new entities or required research branches, add new sub-goals during the run.

### Extraction
Extract useful structured information from already-fetched page content.

### Table extraction
Support:
- cleaned text tables preserved with `|`;
- PDF tables whose column relationships must remain intact.

### Validation
Decide whether a page genuinely resolves a sub-goal.

### Evidence modeling
Store extracted values together with source/evidence metadata.

### Reasoning/synthesis
Compare, summarize, rank, or recommend based only on validated evidence.

### Self-check
Ensure factual claims are supported by evidence from sources visited in the run.

### Logging
Provide structured Person B events, especially planning, validation, extraction, reasoning, and verification.

### Evaluation
Provide a repeatable fixture-based evaluation suite.

### Handoff
Provide a stable public API and detailed Person A integration documentation.

---

# 4. Source of Truth Hierarchy

Implementation decisions must use this priority:

1. Explicit Person A ↔ Person B integration contract in this repository.
2. The latest Person A handoff details.
3. Actual supplied `live.zip` fixture behavior.
4. Original project Build Plan.
5. Reasonable implementation choices required to fill unspecified engineering details.

If source materials conflict, do **not** silently pretend they agree. Build the implementation to handle the observed evidence robustly and document the discrepancy.

---

# 5. Important Ground Truth From the Build Plan

The original Build Plan requires a common pipeline for every task:

```text
Task Understanding
-> Planning
-> Live Navigation
-> Extraction
-> Validation
-> Reasoning/Synthesis
-> Self-Check
-> Final Output
```

Person B is directly responsible for all stages above except live navigation.

The plan specifically requires:

- dynamic growth of sub-goals;
- no fabrication when information cannot be found;
- source URL coverage for factual claims;
- explicit not-found reporting;
- structured logs;
- a fixed evaluation set spanning task types.

---

# 6. Important Ground Truth From Person A's Handoff

## 6.1 Real fixtures are available

Person A provided `live.zip` built from real Banque Misr pages.

Development must stop relying on invented mock-page content when a relevant real fixture exists.

## 6.2 Cleaned `.txt` is the primary HTML-page runtime representation

The handoff states that for HTML pages:

- `.html` is raw server HTML;
- `.txt` is the output after Person A's text cleaner;
- `.txt` is what `fetch_page()` will provide at runtime.

Therefore, Person B's primary extraction path for fetched HTML pages must work on the cleaned `.txt` representation.

Raw HTML may be used only as secondary structure/evidence during fixture development where helpful.

## 6.3 Cleaned text keeps table structure

Tables in cleaned text use pipe delimiters:

```text
Column A | Column B | Column C
--- | --- | ---
Value A | Value B | Value C
```

Person B should implement a deterministic parser for this shape before relying on LLM interpretation.

## 6.4 Navigation boilerplate exists

A large repeated header/navigation region appears before useful content on many pages.

Person B must not mark a sub-goal resolved just because a matching keyword appears in this boilerplate.

## 6.5 Current link record shape

Person A's link dictionaries are:

```python
{
    "label": str,
    "url": str,
    "source": "nav" | "body" | "footer",
    "is_pdf": bool,
    "key": str,
}
```

Person B does not own `key`.

`is_pdf` is only a link hint.

Fetched `content_type` is authoritative.

## 6.6 Person A's requested validation contract

At minimum, Person B must provide:

```python
validate(sub_goal, page_content) -> {
    "resolved": bool,
    "extracted": {...},
    "reason": str,
}
```

Person A's navigation loop uses this decision to continue hopping or stop.

## 6.7 Person A still owns `navigate()`

Person A explicitly identified `navigate(sub_goal)` as Person A's unfinished side.

That does not block Person B development because real fixtures can simulate the page passed into `validate()`.

---

# 7. Fixture Inventory

The supplied fixture package includes real material for:

1. homepage;
2. Cards category;
3. Credit Cards list;
4. Classic Credit Card detail;
5. Accounts & Deposits;
6. Consumer Loans;
7. Fees hub;
8. usage/limits/fees PDF;
9. image-only ATM guide PDF;
10. manifest metadata.

These fixtures cover category pages, lists, product details, tables, multi-product PDF data, and unreadable PDF failure behavior.

---

# 8. Known Fixture/Source Anomalies That Must Become Tests

## 8.1 Misleading page title

The Accounts & Deposits cleaned page begins with a title resembling:

```text
Banquemisr - Classic Credit Card
```

while its body is Accounts & Deposits content.

Implication:

> Page relevance/validation must never depend on title alone.

Use:
- body headings;
- breadcrumb-like body text;
- structured extracted evidence;
- sub-goal semantic fit;
- source URL when available.

## 8.2 Card-fee location discrepancy

The handoff states that card fees are not on product pages and that the cross-card fee PDF is critical.

However, the supplied Classic Credit Card cleaned fixture contains a `Fees and charges` table with issuance, renewal, interest, withdrawal, and other values.

Implementation rule:

> Do not hardcode "fees only live in PDF" or "fees always live on product page."

Instead:
- extract fees wherever evidence exists;
- preserve source URL/evidence per value;
- use the cross-card PDF when the task requires broad/card-to-card fee coverage;
- tolerate duplicate/overlapping fields from multiple sources;
- never merge conflicting values silently.

## 8.3 PDF flat-text alignment problem

The cross-card fee PDF's flat text can lose column alignment.

Person A's handoff explicitly recommends table extraction.

Implementation rule:

> If a PDF task depends on column relationships, use PDF table matrices/raw bytes rather than flattened text alone.

## 8.4 Image-only PDF

The ATM activation guide is image-only and yields no text.

Required behavior:
- do not crash;
- do not return resolved success with empty extraction;
- return an explicit insufficient/unreadable evidence reason.

## 8.5 PDF link extensions are unreliable

Some PDFs are served through `.ashx`; some media links are not PDFs.

Actual fetched `content_type` must override `is_pdf`.

---

# 9. Core Architectural Principles

## 9.1 Portable package

Person B must be importable as one package, independent of Person A internals.

Recommended top-level package:

```text
src/person_b/
```

## 9.2 Stable public facade

Person A should integrate through a small `person_b` public API rather than importing deep internal modules.

Recommended facade responsibilities:

```python
plan_task(...)
next_pending_sub_goal(...)
expand_plan(...)
apply_validation(...)
mark_not_available(...)
validate(...)
extract_content(...)
compare_items(...)
synthesize(...)
validate_answer(...)
finalize(...)
```

Not every name is mandatory, but the final API must be small, documented, and stable.

The minimum `validate(sub_goal, page_content)` contract is mandatory.

## 9.3 Typed internals, serialization-friendly boundary

Use typed internal models for correctness.

Public results that cross into Person A should be easy to serialize to dictionaries/JSON.

Recommended internal models:

- `Goal`
- `SubGoal`
- `Plan`
- `PageContext`
- `Evidence`
- `ExtractedField`
- `ExtractionResult`
- `ValidationResult`
- `ComparisonResult`
- `Claim`
- `ClaimCheck`
- `SynthesisResult`
- `FinalAnswer`

## 9.4 Evidence-first design

The core unit is not merely:

```python
{"issuance_fee": "250 EGP"}
```

It should conceptually preserve:

```python
{
    "field": "issuance_fee",
    "value": "250 EGP",
    "source_url": "...",
    "evidence_text": "Issuance | EGP 250",
    "location": {
        "type": "text_table",
        "table": "Fees and charges",
        "row": "Issuance"
    },
    "confidence": ...
}
```

Exact schema can differ, but source/evidence traceability is mandatory.

---

# 10. Proposed Repository Architecture

Task 01 should create a structure close to:

```text
.
├── README.md
├── AGENT_START_HERE.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── docs/
│   ├── MASTER_PLAN.md
│   ├── contracts/
│   │   └── PERSON_A_PERSON_B_CONTRACT.md
│   └── tasks/
│       ├── TASK_01_FOUNDATION_AND_CONTRACTS.md
│       ├── TASK_02_PLANNING_AND_EXTRACTION.md
│       ├── TASK_03_VALIDATION_REASONING_VERIFICATION.md
│       └── TASK_04_EVALUATION_HARDENING_RELEASE.md
│
├── fixtures/
│   └── live/
│       └── <supplied fixtures>
│
├── src/
│   └── person_b/
│       ├── __init__.py
│       ├── api.py
│       ├── config.py
│       ├── errors.py
│       ├── models.py
│       │
│       ├── planning/
│       │   ├── __init__.py
│       │   ├── planner.py
│       │   └── expansion.py
│       │
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── extractor.py
│       │   ├── text_tables.py
│       │   ├── pdf_tables.py
│       │   └── normalization.py
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   └── validator.py
│       │
│       ├── reasoning/
│       │   ├── __init__.py
│       │   ├── compare.py
│       │   ├── recommend.py
│       │   └── synthesize.py
│       │
│       ├── verification/
│       │   ├── __init__.py
│       │   ├── attribution.py
│       │   └── self_check.py
│       │
│       ├── logging/
│       │   ├── __init__.py
│       │   └── jsonl_logger.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   └── metrics.py
│       │
│       └── adapters/
│           ├── __init__.py
│           ├── reasoning_backend.py
│           └── fixture_loader.py
│
└── tests/
    ├── unit/
    ├── integration/
    └── evaluation/
```

Anti-Gravity may make small layout changes if they materially improve maintainability, but:

- do not create Person A modules;
- do not bury the public facade;
- do not duplicate responsibilities across many unnecessary files;
- update the docs if the final structure differs.

---

# 11. Runtime Reasoning Backend Strategy

The original project expects an LLM-driven planner/orchestrator/reasoning layer.

However, this plan only specifies the **coding model** (Gemini 3.7 Flash/high reasoning). It does not establish that the finished application must use the same runtime model.

Therefore:

1. define a narrow runtime reasoning abstraction/protocol;
2. inject/configure it;
3. keep unit tests deterministic using a fake/stub backend;
4. do not make live LLM calls in normal unit tests;
5. if the existing repository already defines a runtime provider, reuse it;
6. otherwise leave the provider selection explicit and documented rather than coupling all Person B code to one SDK.

A real runtime adapter may be implemented if the project configuration provides enough information, but the core Person B algorithms/models must remain provider-agnostic.

---

# 12. Planning Requirements

## 12.1 Task understanding

Convert raw request into a normalized goal that preserves user intent.

Do not add unsupported requirements.

## 12.2 Initial sub-goals

Create the smallest useful set of initial sub-goals.

For a discovery/comparison task, do not eagerly invent one sub-goal per product before the site reveals those products.

## 12.3 Dynamic expansion

After a validated extraction reveals entities or missing dimensions, expand.

Example:

```text
Find all credit cards
-> validated list of card names
-> add one research sub-goal per card or per missing evidence group
```

Expansion must be:
- deterministic enough to deduplicate;
- bounded;
- stateful;
- aware of already-resolved/pending sub-goals.

## 12.4 No task-type hardcoding

Allowed:
- general prompts/templates for lookup/comparison/recommendation;
- generic sub-goal patterns.

Not allowed:
- `if task == credit_cards: go to X`;
- fixed list of Banque Misr cards in source;
- per-task hardcoded answer content.

---

# 13. Extraction Requirements

## 13.1 Input

Primary input is content already fetched/cleaned by Person A.

At minimum:

```python
extract_content(page_content, target_fields)
```

Enriched context should support:
- `source_url`;
- `content_type`;
- status;
- optional raw PDF bytes/path;
- optional pre-extracted PDF tables.

## 13.2 Cleaned-text preprocessing

Person B should:
- preserve useful headings;
- recognize repeated boilerplate;
- preserve original evidence text;
- avoid destructively deleting content without retaining raw input.

## 13.3 Pipe-table parser

Implement deterministic recognition of pipe-delimited tables.

Handle:
- optional separator rows;
- uneven whitespace;
- cells containing punctuation;
- short two-column tables;
- multiple tables on one page;
- surrounding prose.

## 13.4 PDF tables

Use a PDF table extraction library such as `pdfplumber` when raw PDF bytes/path are available.

Iterate across **all pages and all tables**.

Normalize:
- multiline cells;
- `None` cells;
- repeated/split headers;
- whitespace.

Do not rely on `extract_text()` for table-critical cross-card mapping.

## 13.5 Empty PDF

If no text/tables are extractable:

```text
Extraction status = unreadable/insufficient
Validation must not resolve the sub-goal from that PDF alone.
```

## 13.6 Generic structured extraction

The extractor must accept target fields/intent rather than being tied to:
- fee;
- benefit;
- eligibility only.

It should support arbitrary requested fields from banking pages.

---

# 14. Validation Requirements

Validation answers:

> Does this page actually answer the current sub-goal with usable evidence?

It is not merely a keyword matcher.

Signals may include:

- source URL path;
- body heading match;
- structured extracted fields;
- semantic relevance;
- whether required fields/entities are present;
- whether content is only nav/footer boilerplate;
- whether the page is an unreadable PDF;
- whether evidence is partial.

Required result:

```python
{
    "resolved": bool,
    "extracted": {...},
    "reason": str,
}
```

Recommended additional fields:

```python
{
    "status": "resolved" | "partial" | "unresolved" | "unreadable",
    "evidence": [...],
    "missing_fields": [...],
    "source_url": "...",
    "confidence": ...
}
```

The minimum three fields remain mandatory.

---

# 15. Partial Evidence Policy

A page can be relevant without fully resolving the sub-goal.

Example:

```text
Sub-goal:
"Get fees and benefits for Card X"

Page:
contains benefits but no fee evidence.
```

Possible behavior:

- return partial/unresolved for the whole combined sub-goal;
- preserve the benefits evidence;
- let navigation continue for fee evidence.

Do not throw away valid partial evidence.

The planner may also split a combined sub-goal if that is clearer.

---

# 16. Reasoning and Comparison

`compare_items(items)` must:

- operate only on validated structured records;
- preserve missing fields;
- never fill absent values from model prior knowledge;
- retain per-item/per-field evidence links;
- make differences explicit.

Recommendations must:

1. infer criteria from the user's stated need;
2. score/reason only over available evidence;
3. explain tradeoffs;
4. distinguish fact from recommendation judgment;
5. expose missing evidence that could change the recommendation.

---

# 17. Synthesis Requirements

Synthesis combines validated results into:

- direct answer;
- comparison;
- summary;
- recommendation;
- multi-step research result.

Every factual claim created by synthesis must be convertible into a `Claim` object containing enough information for source verification.

Synthesis should not be the first place source attribution is added. Evidence lineage must already exist in extracted records.

---

# 18. Self-Check / Hallucination Guardrail

Before final output:

```python
validate_answer(claims, visited_pages)
```

must verify that every factual claim:

- maps to evidence;
- evidence comes from a page/source from this run;
- source URL is known where final sourcing is required;
- the claim does not overstate the evidence.

Unsupported claims must fail verification.

Finalization must either:
- remove them;
- narrow them;
- or explicitly report that the information was not found.

---

# 19. Source Attribution Rules

The final system should be able to return:

```text
Answer
+
exact source URLs used
+
explicit gaps/not-found items
```

Person B therefore needs URL-aware evidence records.

The minimal two-argument validation contract remains valid for Person A compatibility, but production integration should pass `source_url`.

---

# 20. Logging

Person B logging should be structured, preferably JSONL.

Log event types should include:

- task understood;
- plan created;
- sub-goal added;
- extraction attempted;
- extraction succeeded/failed;
- validation resolved/partial/unresolved;
- comparison performed;
- synthesis performed;
- claim verification pass/fail;
- not-found decision;
- error.

Do not log:
- secrets/API keys;
- entire huge raw pages unnecessarily;
- raw PDF bytes.

Every event should have:
- timestamp;
- run/task ID when available;
- event type;
- sub-goal ID when relevant;
- concise structured details.

---

# 21. Evaluation Plan

The evaluation suite must use the same Person B code path as normal operation.

Required categories:

1. **lookup**
2. **comparison**
3. **recommendation**
4. **multi-hop/dynamic expansion**
5. **trap / not available**
6. **unreadable PDF**
7. **misleading title / boilerplate false-positive defense**

Suggested metrics:

- extraction correctness;
- sub-goal resolution correctness;
- false-positive validation rate;
- required-field coverage;
- claim support rate;
- source attribution rate;
- correct not-found rate;
- plan expansion correctness.

No evaluation test should require live Banque Misr requests.

---

# 22. Error Model

Normal content failures should be represented as structured results, not unhandled exceptions.

Examples:

- empty page;
- irrelevant page;
- malformed table;
- insufficient fields;
- unreadable PDF;
- missing URL;
- no table structure for table-critical PDF;
- invalid model output that can be retried/normalized.

Programmer/configuration errors may raise typed exceptions.

---

# 23. Security and Robustness

Treat page content as untrusted text.

Do not execute code or instructions embedded in page content.

If LLM prompts consume website text:
- clearly delimit it as data;
- tell the model not to follow instructions from retrieved content;
- use structured output validation;
- cap input sizes;
- preserve deterministic parsers for tables.

---

# 24. Performance Expectations

Person B should avoid adding unnecessary live latency.

- no live HTTP inside Person B;
- parse tables locally;
- avoid repeated parsing of identical content in one call;
- reuse normalized extraction result in validation/synthesis;
- do not call the runtime LLM when deterministic extraction is sufficient unless reasoning is actually needed.

---

# 25. Test Strategy

## Unit tests

Test individual:
- table parser;
- PDF table normalizer;
- models;
- planner expansion/deduplication;
- validation heuristics/contracts;
- source attribution;
- self-check;
- logging.

## Fixture integration tests

Use real fixture files to test:
- Cards category;
- Credit Cards list;
- Classic details;
- Accounts & Deposits title anomaly;
- Consumer Loans;
- Fees hub;
- fee PDF;
- image-only PDF.

## Contract tests

Prove:

```python
validate(sub_goal, page_content)
```

works without additional arguments.

Also test enriched calls.

## No-live-network rule

Tests must fail or be clearly separated if they attempt live Banque Misr access.

---

# 26. Implementation Tasks

We deliberately keep the implementation in **four large tasks**.

## Task 01 — Foundation and Contracts

Create:
- package structure;
- configuration;
- models;
- public API skeleton;
- runtime reasoning interface;
- fixture loader;
- test scaffolding;
- dependency configuration;
- contract tests.

No deep planner/extraction logic yet.

## Task 02 — Planning and Extraction

Implement:
- task understanding;
- plan/sub-goal lifecycle;
- dynamic expansion;
- cleaned-text preprocessing;
- pipe-table parser;
- PDF table parser;
- extraction pipeline;
- evidence records;
- relevant fixture tests.

## Task 03 — Validation, Reasoning, Verification

Implement:
- `validate()`;
- partial evidence behavior;
- comparison;
- recommendation reasoning;
- synthesis;
- claims;
- source attribution;
- `validate_answer()`;
- finalization;
- false-positive/unreadable handling.

## Task 04 — Evaluation, Hardening, Release

Implement/finalize:
- JSONL logging;
- evaluation runner and metrics;
- complete regression suite;
- integration smoke adapter/example;
- packaging cleanup;
- dependency cleanup;
- public API stabilization;
- README integration accuracy;
- final handoff checklist.

---

# 27. Git/Repository Policy

This repository belongs only to Person B.

Do not create Person A branches or copy Person A's source.

Recommended commits:

```text
docs: add person B master plan and task specifications
feat: establish person B package and contracts
feat: implement planner and extraction
feat: implement validation reasoning and verification
test: add evaluation and hardening
docs: finalize person A integration handoff
```

Person A will later clone/copy this repository into Person A's own environment.

---

# 28. Public Contract Compatibility

The following must remain possible:

```python
result = validate(sub_goal, page_content)
assert "resolved" in result
assert "extracted" in result
assert "reason" in result
```

Enrichment must be additive.

Preferred optional context:

```python
validate(
    sub_goal,
    page_content,
    source_url=None,
    content_type=None,
    status_code=None,
    pdf_bytes=None,
    pdf_path=None,
    pdf_tables=None,
    metadata=None,
)
```

Exact optional names may change if a cleaner design is chosen, but README and contract docs must be updated together.

---

# 29. Definition of Done — Entire Person B

Person B is done only when all are true:

## Architecture
- [x] clear standalone `person_b` package exists;
- [x] no dependency on Person A internals;
- [x] public API is small and documented.

## Planning
- [x] user task can become a structured goal;
- [x] initial sub-goals are generated;
- [x] plan can grow dynamically;
- [x] duplicate sub-goals are prevented.

## Extraction
- [x] cleaned text is supported;
- [x] pipe tables are parsed;
- [x] PDF tables are supported;
- [x] all pages/all tables are iterated where appropriate;
- [x] empty/image-only PDF is handled safely;
- [x] evidence lineage is retained.

## Validation
- [x] exact Person A minimum contract works;
- [x] body evidence matters more than title alone;
- [x] nav boilerplate does not create false resolution;
- [x] partial evidence is preserved;
- [x] unreadable content returns unresolved.

## Reasoning
- [x] compare works on structured records;
- [x] recommendation reasoning is evidence-bound;
- [x] synthesis handles all target task types;
- [x] missing values remain missing.

## Verification
- [x] claims are explicit;
- [x] every factual claim can be source checked;
- [x] unsupported claims fail;
- [x] not-found is explicit.

## Logging/Evaluation
- [x] structured logs exist;
- [x] evaluation categories are present;
- [x] fixture regression suite passes;
- [x] no normal test needs live site access.

## Handoff
- [x] README contains current detailed Person A integration steps;
- [x] contract documentation is current;
- [x] dependencies are declared;
- [x] Person A can run a fixture smoke test;
- [x] final API examples in README use real implemented names.

---

# 30. Final Instruction to Anti-Gravity

Do not attempt to implement the entire plan in one uncontrolled pass.

For each task:

1. read all required docs;
2. inspect current code;
3. implement only the current task;
4. run tests;
5. fix failures;
6. report changes;
7. stop.

The goal is speed **without sacrificing compatibility or correctness**.
