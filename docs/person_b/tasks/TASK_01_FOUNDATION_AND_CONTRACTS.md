# TASK 01 — Foundation, Architecture, and Contracts

## Anti-Gravity Execution Profile

- Model: Gemini 3.7 Flash
- Reasoning: High / maximum available
- Scope: Person B repository only

---

# Objective

Create the complete **foundation** for Person B without implementing the deep planning/extraction/reasoning behavior yet.

After this task, the repository must have:

- a clean Python package;
- stable integration boundaries;
- typed data models;
- dependency/config setup;
- fixture access;
- test scaffolding;
- public API placeholders that fail safely where not yet implemented;
- mandatory `validate(sub_goal, page_content)` compatibility established by tests.

This task determines the architecture that later tasks fill in.

---

# Mandatory Preflight

Before coding, read completely:

1. `AGENT_START_HERE.md`
2. `docs/MASTER_PLAN.md`
3. `docs/contracts/PERSON_A_PERSON_B_CONTRACT.md`
4. `README.md`
5. this task file

Then inspect:

- all existing files;
- whether `fixtures/live/` exists;
- existing dependency configuration;
- any existing package naming conventions.

Do not delete supplied docs/fixtures.

---

# Scope

## In scope

- project/package structure;
- `pyproject.toml` or equivalent;
- `.gitignore`;
- `.env.example`;
- typed models;
- Person B errors;
- public API facade;
- runtime reasoning backend protocol/interface;
- fixture loader;
- logging interface stub or basic event types;
- unit/integration test directories;
- basic contract tests;
- package import smoke test.

## Out of scope

Do not implement:

- Person A live fetching;
- Person A navigation;
- link extraction;
- real planner behavior;
- dynamic plan expansion logic beyond model/method scaffolding;
- deep extraction;
- PDF table extraction;
- synthesis;
- recommendation;
- full evaluation.

---

# Required Package Structure

Use the Master Plan architecture as the default.

At minimum the final package should clearly separate:

```text
planning/
extraction/
validation/
reasoning/
verification/
logging/
evaluation/
adapters/
```

Avoid excessive micro-files.

---

# Public API Requirement

Create a public facade through `person_b/__init__.py` and/or `person_b/api.py`.

The long-term API must be able to expose planning, validation, reasoning, and verification.

For this task, the critical requirement is that this import path exists:

```python
from person_b import validate
```

and that this call is accepted:

```python
result = validate(sub_goal, page_content)
```

Because deep validation is not implemented yet, the placeholder must:

- not pretend the sub-goal is resolved;
- return a safe unresolved structure;
- include the required keys.

Example acceptable foundation behavior:

```python
{
    "resolved": False,
    "extracted": {},
    "reason": "Validation engine not implemented yet."
}
```

Do not return always-true.

---

# Data Models

Implement typed models sufficient for later tasks.

Recommended models and fields:

## Goal

- `id`
- `statement`
- `original_task`

## SubGoal

- `id`
- `question`
- `status`
- `target_fields`
- `parent_id`
- `depends_on`
- `created_from`
- optional metadata

Statuses should cover at least:

```text
pending
in_progress
resolved
partial
not_available
failed
```

## Plan

- plan ID
- goal
- ordered sub-goals
- optional metadata/version

## PageContext

Must be able to represent:

- page content string;
- source URL;
- content type;
- status code;
- optional PDF bytes/path/tables;
- metadata.

Do not require all fields.

## Evidence

At minimum:

- field or claim target;
- value;
- source URL optional;
- evidence text optional;
- extraction location/type;
- confidence optional.

## ExtractionResult

- status
- extracted structured data
- evidence
- missing fields
- warnings
- parser metadata

## ValidationResult

Minimum serialized fields:

```text
resolved
extracted
reason
```

Recommended:
- status
- evidence
- missing_fields
- source_url
- warnings

## Claim / ClaimCheck

Needed by later self-check.

---

# Serialization Boundary

Ensure models can be converted to plain dictionaries cleanly.

Person A should not need access to internal classes to use the package.

If using Pydantic:
- use current supported Pydantic APIs;
- do not expose Pydantic-only behavior as a hard integration requirement.

---

# Reasoning Backend Interface

Create a provider-agnostic protocol/abstract interface for runtime LLM reasoning.

It should support the types of operations later needed for:

- task understanding/planning;
- structured extraction fallback;
- synthesis/recommendation.

Requirements:

- injected/configurable;
- unit-testable with a fake implementation;
- no live LLM call during ordinary tests;
- no API key committed.

Do not equate Anti-Gravity's coding model with runtime model automatically.

---

# Configuration

Create settings for:

- optional runtime reasoning provider;
- optional runtime model;
- logging path;
- strict verification toggle if useful;
- parser limits;
- optional confidence threshold if design uses one.

Do not add Person A's navigation limits as Person B-owned configuration unless needed only for local evaluation metadata.

---

# Fixture Loader

Implement a small adapter for development/tests that can load:

- cleaned `.txt`;
- raw `.html` when requested;
- PDFs as bytes/path;
- `manifest.json`.

It must not pretend to be Person A navigation.

Purpose:

```text
fixture file -> PageContext
```

not:

```text
sub-goal -> crawl website
```

---

# Dependency Policy

Keep dependencies minimal.

Likely categories:
- schema/validation library if used;
- PDF table library;
- test framework;
- optional runtime reasoning SDK only if explicitly configured.

Do not add HTTP crawling libraries just for Person B.

---

# Testing Required

At minimum add tests for:

1. package imports;
2. data model serialization;
3. safe placeholder validation;
4. exact minimum validation contract;
5. enriched validation arguments accepted by the public wrapper;
6. fixture loader can locate/load the real fixtures when present;
7. no live requests are made.

---

# Documentation Update

If your actual structure/API differs from the Master Plan's proposed structure:

- update `README.md`;
- update `docs/MASTER_PLAN.md`;
- update `docs/contracts/PERSON_A_PERSON_B_CONTRACT.md`;

in the same task so documentation never knowingly lies.

---

# Quality Checks

Run at minimum:

```bash
python -m pytest -q
python -m compileall src
```

If linting is configured:

```bash
ruff check .
```

Fix failures before finishing.

---

# Deliverable Report

At completion report:

1. final tree;
2. package name/import path;
3. public API skeleton;
4. models created;
5. dependency files created;
6. fixture loader behavior;
7. tests/checks and results;
8. deviations from Master Plan;
9. blockers.

---

# Definition of Done

- [ ] standalone package exists;
- [ ] no Person A code was implemented;
- [ ] mandatory minimum `validate()` call works;
- [ ] `validate()` does not falsely resolve before Task 03;
- [ ] typed models exist;
- [ ] serialization works;
- [ ] reasoning backend is provider-agnostic;
- [ ] fixture loading works;
- [ ] tests pass;
- [ ] docs remain accurate.
