# TASK 03 — Validation, Reasoning, Synthesis, and Verification

## Anti-Gravity Execution Profile

- Model: Gemini 3.7 Flash
- Reasoning: High / maximum available
- Scope: Person B validation/reasoning/verification

---

# Objective

Finish Person B's core intelligence layer:

- sub-goal validation;
- partial evidence behavior;
- comparison;
- recommendation;
- synthesis;
- explicit claims;
- source attribution;
- `validate_answer()` hallucination guardrail;
- final answer assembly.

At the end of this task, Person B should be functionally complete except for final logging/evaluation/release hardening.

---

# Mandatory Preflight

Read all master/contract docs plus Tasks 01 and 02.

Run the complete existing test suite before starting.

Inspect:
- planner output;
- extraction result models;
- evidence model;
- public API facade.

Do not redesign working Task 02 extraction without a demonstrated need.

---

# Part A — Implement `validate()`

Mandatory minimum call:

```python
validate(sub_goal, page_content)
```

Mandatory output keys:

```python
{
    "resolved": bool,
    "extracted": {...},
    "reason": str,
}
```

Enriched keyword context may be supported.

`validate()` should:

1. normalize input into `PageContext`;
2. run/reuse extraction;
3. judge relevance;
4. judge whether requested evidence is sufficient;
5. preserve partial evidence;
6. return structured status/reason.

---

# Part B — Validation Must Not Be a Keyword Check

Bad behavior:

```text
Sub-goal: Accounts
Page header boilerplate contains "Accounts"
=> resolved
```

Required behavior:

- distinguish body content from repeated navigation;
- check structured extraction;
- consider body headings/breadcrumb-like context;
- optionally use source URL;
- use semantic reasoning only after deterministic signals are prepared.

---

# Part C — Misleading Title Regression

The Accounts & Deposits fixture begins with a misleading Classic Credit Card title.

Add a regression test showing:

- validation for Accounts & Deposits can still succeed;
- validation does not classify the page as a credit-card detail page merely from title.

---

# Part D — Partial Evidence

Support:

```text
resolved=False
status=partial
extracted=<valid evidence found so far>
missing_fields=<still needed>
```

Example:

```text
Need fees + benefits
Page gives benefits only
```

Person A should continue navigation while retaining Person B's partial evidence.

---

# Part E — Unreadable PDF Validation

If extraction marks an image-only PDF unreadable:

```python
{
    "resolved": False,
    "status": "unreadable",
    "extracted": {},
    "reason": ...
}
```

No LLM may turn the file name into an invented answer.

---

# Part F — `compare_items(items)`

Implement comparison over validated structured records.

Requirements:

- stable rows/fields;
- explicit missing values;
- per-field evidence preserved;
- no invented normalization that changes meaning;
- ability to render/serialize a comparison structure.

When comparing money/rates/limits:
- preserve original textual value;
- optional normalized numeric representation can be added separately;
- do not discard units/conditions.

---

# Part G — Recommendation Reasoning

Recommendations should:

1. derive criteria from the user's stated need;
2. map evidence to each criterion;
3. identify tradeoffs;
4. explain uncertainty/missing data;
5. never promote an item based on unsupported facts.

Separate:

- factual evidence;
- reasoning judgment.

For example:

```text
Fact: Card A has X documented fee.
Judgment: Therefore Card A may be preferable for a user prioritizing lower issuance cost.
```

The judgment is allowed only if its premise is supported.

---

# Part H — Synthesis

Implement a synthesis layer supporting:

- lookup;
- list/discovery;
- comparison;
- recommendation;
- multi-step/process research.

Synthesis input should be validated plan/results, not raw random pages.

Synthesis output should include:

- answer content/structure;
- claims;
- sources;
- gaps/not-found items;
- optional reasoning notes that are safe for user output.

---

# Part I — Claim Model

Every factual statement intended for final output should be represented as a claim.

Recommended:

```python
Claim(
    id=...,
    text=...,
    evidence_ids=[...],
    source_urls=[...],
    claim_type="fact" | "reasoned_conclusion",
)
```

A reasoned conclusion still needs its factual premises connected.

---

# Part J — Source Attribution

Implement source attribution that:

- deduplicates URLs for final source list;
- preserves multiple sources when needed;
- rejects evidence with a source URL that was not visited in the current run when strict verification is enabled;
- can explain which claim maps to which evidence.

---

# Part K — `validate_answer()`

Conceptual API:

```python
validate_answer(claims, visited_pages)
```

For each claim:

- support found?
- source visited?
- evidence adequate?
- claim broader than evidence?
- contradictory evidence?

Return a pass/fail/repair-needed result per claim.

Do not merely ask an LLM "is this supported?" without deterministic source/evidence checks.

---

# Part L — Finalization

Implement final assembly that removes or repairs unsupported claims.

Final output model should support:

```python
{
    "answer": ...,
    "sources": [...],
    "not_found": [...],
    "verification": ...
}
```

Exact schema may differ.

The final answer must never silently include a failed claim.

---

# Part M — Not-found Policy

When a sub-goal cannot be resolved from provided/visited evidence:

Use explicit language/state equivalent to:

```text
not found on the Banque Misr website for this item
```

Do not infer a value from common banking knowledge.

---

# Part N — Conflicting Evidence

If two visited sources disagree:

- surface the conflict internally;
- do not silently choose one;
- prefer a cautious final answer;
- include both source references when useful;
- indicate that the website materials conflict if the conflict cannot be resolved.

This behavior is more important than producing a complete-looking table.

---

# Part O — Integration-facing Methods

By the end of Task 03, the public API should be close to final.

Ensure Person A can perform the conceptual flow:

```python
plan = plan_task(user_task)

sub_goal = next_pending_sub_goal(plan)

verdict = validate(
    sub_goal,
    page.content,
    source_url=page.url,
    content_type=page.content_type,
)

plan = apply_validation(plan, sub_goal, verdict)
plan = expand_plan(plan, verdict)

draft = synthesize(user_task, plan)
checked = validate_answer(draft_claims, visited_pages)
final = finalize(draft, checked)
```

If actual names differ, update all docs immediately.

---

# Required Tests

## Validation

- correct card-list page resolves card discovery;
- irrelevant page does not resolve;
- nav boilerplate does not falsely resolve;
- misleading title does not override body evidence;
- partial result is retained;
- image-only PDF is unresolved/unreadable.

## Comparison

- missing fields stay missing;
- evidence remains attached;
- source conflicts are not silently overwritten.

## Recommendation

- recommendation changes when evidence changes;
- no unsupported criterion is invented.

## Verification

- supported claim passes;
- unsupported claim fails;
- claim with unvisited source fails in strict mode;
- duplicate sources dedupe;
- overbroad claim fails/narrows.

## End-to-end fixture flow without live navigation

Simulate:

```text
task
-> plan
-> manually supply fixture(s)
-> validate
-> expand
-> synthesize
-> self-check
-> final
```

---

# Quality Checks

Run all tests plus:

```bash
python -m compileall src
```

and configured linting.

---

# Deliverable Report

Report:

1. final validation semantics;
2. public API;
3. partial behavior;
4. comparison/recommendation design;
5. claim/evidence verification design;
6. end-to-end fixture scenario results;
7. test results;
8. remaining hardening items for Task 04.

---

# Definition of Done

- [ ] minimum validation contract works;
- [ ] enriched validation works;
- [ ] false-positive defenses work;
- [ ] partial evidence works;
- [ ] unreadable PDF works;
- [ ] comparison works;
- [ ] recommendation is evidence-bound;
- [ ] synthesis works across target task types;
- [ ] claims are explicit;
- [ ] source attribution works;
- [ ] `validate_answer()` rejects unsupported claims;
- [ ] finalization removes/repairs failed claims;
- [ ] end-to-end fixture test passes.
