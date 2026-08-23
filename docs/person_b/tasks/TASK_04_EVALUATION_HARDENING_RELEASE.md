# TASK 04 — Logging, Evaluation, Hardening, Release, and Handoff

## Anti-Gravity Execution Profile

- Model: Gemini 3.7 Flash
- Reasoning: High / maximum available
- Scope: finish Person B and prepare Person A handoff

---

# Objective

Turn the functionally complete Person B implementation into a **release-ready handoff**.

This task covers:

- structured logging;
- evaluation;
- regression hardening;
- dependency/package cleanup;
- final public API stabilization;
- README integration accuracy;
- Person A smoke-integration example;
- final Definition of Done audit.

At the end, Person B should be ready to send to Person A as a repository link.

---

# Mandatory Preflight

Read:

1. `AGENT_START_HERE.md`
2. `docs/MASTER_PLAN.md`
3. `docs/contracts/PERSON_A_PERSON_B_CONTRACT.md`
4. `README.md`
5. Tasks 01–03
6. this task file

Run all current tests before changing code.

Inspect all public exports and compare them against README examples.

---

# Part A — Structured Logging

Implement/finalize Person B JSONL logging.

Required event categories:

- task understanding;
- plan creation;
- sub-goal creation;
- plan expansion;
- extraction attempt/result;
- PDF parse result;
- validation result;
- partial evidence;
- comparison;
- recommendation;
- synthesis;
- claim verification;
- not-found;
- error.

Each event should have:

- UTC timestamp;
- run/task ID where provided;
- event type;
- sub-goal ID where relevant;
- concise structured payload.

Do not log:
- secrets;
- API keys;
- PDF bytes;
- giant raw page bodies by default.

Logging should be optional/configurable and must not break core behavior if a log file cannot be written unless strict mode explicitly requires it.

---

# Part B — Evaluation Dataset/Cases

Create a fixed evaluation suite covering at least:

## 1. Lookup
Example class:
- identify available product/category information from one relevant page.

## 2. Comparison
Example class:
- compare card fields across structured evidence.

## 3. Recommendation
Example class:
- choose based on user-stated criteria and available evidence.

## 4. Dynamic/multi-step
Example class:
- list discovered entities then expand research sub-goals.

## 5. Trap / not available
Ask for a field not present in supplied evidence and ensure no fabrication.

## 6. Unreadable PDF
Use image-only PDF and ensure correct not-available/unreadable behavior.

## 7. Misleading title
Use Accounts & Deposits fixture and ensure body evidence wins.

## 8. Boilerplate false-positive
Construct/use a sub-goal term that appears in nav/footer but not real page body answer.

---

# Part C — Metrics

Implement clear evaluation metrics, such as:

- validation accuracy;
- false-positive validation count/rate;
- field extraction correctness;
- field coverage;
- not-found correctness;
- claim support rate;
- source attribution rate;
- unsupported claim rate;
- dynamic expansion success.

Metrics must be understandable in a grading/demo context.

---

# Part D — Deterministic Evaluation

Evaluation should not depend on live website state.

If a runtime LLM is needed:
- support a deterministic fixture/fake backend for regression;
- optionally provide a separately marked runtime/integration mode.

A passing test suite should not require API credentials.

---

# Part E — Robustness Hardening

Audit and add tests for:

- empty string;
- whitespace-only content;
- unexpected encoding artifacts;
- Arabic + English mixed text;
- duplicated page sections;
- multiple similar headings;
- malformed pipe rows;
- extremely wide PDF tables;
- empty PDF;
- missing URL;
- unknown content type;
- content type contradicting `is_pdf` hint;
- duplicate sub-goals;
- duplicate evidence;
- conflicting evidence;
- invalid reasoning-backend structured output;
- missing optional configuration;
- logger path failure.

Do not overengineer exotic cases unrelated to the supplied website behavior.

---

# Part F — Performance/Context Hardening

Verify:

- large raw pages are not duplicated excessively in memory/logs;
- boilerplate-focused text is available for reasoning;
- evidence snippets stay small;
- PDF tables are parsed once per page/context when possible;
- deterministic extraction happens before expensive model reasoning.

---

# Part G — Public API Freeze

Before release:

1. list every exported public function/type;
2. remove accidental internal exports;
3. ensure names are intuitive;
4. ensure minimum validation contract still works;
5. ensure public return values are serialization-friendly.

After this point, treat the API as versioned.

---

# Part H — README Integration Rewrite

This is a mandatory deliverable.

Read the entire `README.md` and update it to match the **actual implementation**, not the original plan.

The final README must contain:

1. what Person B is;
2. Person A vs Person B ownership;
3. installation;
4. exact package copy/install options;
5. dependency setup;
6. runtime reasoning configuration;
7. exact imports;
8. exact planning example;
9. exact `validate()` example;
10. unresolved/partial handling;
11. dynamic expansion flow;
12. PDF raw bytes/tables integration;
13. image-only PDF behavior;
14. `content_type` vs `is_pdf` rule;
15. source URL requirements;
16. exact synthesis/verification calls;
17. end-to-end Person A loop pseudocode using real final names;
18. fixture smoke test;
19. error behavior;
20. final integration checklist.

Remove any stale placeholder API names.

---

# Part I — Contract Documentation Audit

Update:

```text
docs/contracts/PERSON_A_PERSON_B_CONTRACT.md
```

so it reflects:
- exact function signatures;
- exact required/optional fields;
- exact result schemas;
- version information;
- PDF behavior;
- errors;
- examples.

The contract must be enough for Person A to integrate without reading internal modules.

---

# Part J — Handoff Smoke Example

Provide either:
- a runnable `examples/person_a_integration_smoke.py`, or
- a test that serves the same purpose.

It must simulate Person A by loading fixtures and passing them into Person B.

It must **not** implement real navigation.

The example should demonstrate:
- plan;
- supplied page;
- validate;
- continue/stop signal;
- plan update/expansion;
- synthesis;
- self-check;
- final output.

---

# Part K — Release Checklist

Audit the entire Master Plan Definition of Done.

Check every box based on real tests/code.

Do not claim completion for unimplemented items.

If something remains blocked by an external runtime model/provider decision, isolate it behind the defined adapter and document precisely what Person A/user must configure.

---

# Part L — Final Test Commands

Run:

```bash
python -m pytest -q
python -m compileall src
```

Run linter if configured.

If packaging is configured, also test a clean install in a temporary environment if practical.

At minimum verify:

```python
import person_b
```

works after the documented installation method.

---

# Part M — Final Handoff Report

Create a concise release/handoff note containing:

- version/tag recommendation;
- final public API;
- dependencies;
- all tests run/results;
- known limitations;
- exact Person A integration entry point;
- PDF requirement;
- no-live-network confirmation for tests;
- files Person A actually needs to copy/install.

This can be added to README or a release notes file.

---

# Definition of Done

- [ ] structured logging implemented;
- [ ] evaluation suite implemented;
- [ ] required task categories covered;
- [ ] metrics reported;
- [ ] robustness regression tests added;
- [ ] all tests pass;
- [ ] public API frozen/documented;
- [ ] README uses actual API names;
- [ ] README integration guide is detailed and complete;
- [ ] contract file matches code;
- [ ] smoke integration example works;
- [ ] no Person A live navigation was implemented;
- [ ] dependencies are declared;
- [ ] Person B can be handed off by repository link;
- [ ] Master Plan project-level Definition of Done is satisfied or any exception is explicitly documented.
