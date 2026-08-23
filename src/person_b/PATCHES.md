# Patches applied to the vendored Person B package

Every change to `src/person_b/` beyond the directory rename is listed here.
Each one is a marked block in the source:

```python
# --- PATCH n (vendor) ---
...
# --- END PATCH n ---
```

so `grep -rn "PATCH .* (vendor)" src/person_b/` finds all of them, and a future
drop from Person B can be re-vendored by reapplying this list.

Regression coverage lives in **`tests/test_person_b_patches.py`**, deliberately
outside `tests/person_b_vendor/` so that replacing their suite wholesale cannot
silently drop the guards. Their own 70 tests still pass unmodified.

## Why these were patched here rather than upstream

The bugs were found by probing their code against this repo's `fixtures/live/`,
not by reading their handoff. Person B could not turn them around in time.
Patching a vendored dependency is a cost, so the bar was: fix the root cause
where it is contained to a function, never rewrite a module, and record the
measurement.

## The measurement

`scripts/validator_bench.py` scores the validator against **23 labelled
`(task, page)` pairs** drawn from the real fixtures. Labels describe what each
page actually contains — checked with grep — and were written before any fix
was chosen, so they are not tuned to a threshold.

| | correct | false positives | false negatives |
|---|---|---|---|
| **Before** | 10 / 23 | **9** | 4 |
| **After** | 21 / 23 | **0** | 2 |

A *false positive* is the expensive one: the validator resolves on a page that
cannot answer the question, so navigation stops there and the run reports
success with the wrong page. A *false negative* costs a hop.

### The topic-gate tolerance sweep

The gate asks whether every content word of the question appears on the page.
`_TOPIC_MISS_TOLERANCE` is how many may be missing. Swept over the same 23
cases (`python3 -c "import sys; sys.path[:0]=['scripts','src']; import validator_bench as b; b.sweep()"`):

| tolerance | correct | FP | FN | |
|---|---|---|---|---|
| **0** | 21/23 | **0** | 2 | every content word must appear — **shipped** |
| 1 | 19/23 | 3 | 1 | |
| 2 | 18/23 | 5 | 0 | |
| off | 15/23 | 8 | 0 | the original behaviour |

Loosening buys back the fees-hub false negative and pays three false positives
for it — "does Banque Misr offer student accounts?", "…a cryptocurrency deposit
account?" and "what is the interest rate on a car loan?" all start resolving
again on pages that say nothing about those things. That trade is not worth
making, so tolerance ships at 0 and the false negative below is left standing.

**The stopword list and the tolerance interact.** Re-run the sweep after
touching either.

### What is still wrong, and why it was left

Both remaining false negatives are the same page and the same cause:

| task | page | missing words |
|---|---|---|
| "Where is the schedule of fees and commissions?" | fees hub | `schedule`, `commissions` |
| "Show me the fees and rates document" | fees hub | `document` |

The fees hub says *"Fees and Rates"* and *"Attachments Section"*. It never uses
the words "schedule", "commissions" or "document". **No threshold on literal
token matching can bridge that** — it is a paraphrase problem, and their layer
has no model in it by design. The honest fixes are a synonym/stemming layer or
an LLM, both of which are larger than a vendor patch. Stemming alone would
recover `commissions` → `commission` but not `schedule`, so it would not close
either case.

Practical effect: a fees question phrased in the site's own vocabulary
resolves; one phrased in the user's may cost extra hops and land on `partial`
or `exhausted` rather than `resolved`. It fails in the safe direction.

---

## The patches

### PATCH 1 — `extraction/extractor.py`: mojibake constant

`_NAV_START_MARKERS` was **dead code** — defined, never read; only the footer
half of the intended boilerplate stripping was wired up. Its first entry was
the literal `"ie ??? ????"`, a mangled Arabic phrase that had lost its encoding
before being pasted in as a constant. Removed. Behaviour-preserving.

### PATCH 2 — `planning/planner.py`: invented target fields

`_extract_requested_fields` mapped *any* task containing the word "card" to
required fields `["fees", "benefits"]`. The validator then refused to resolve
until both were present, so "Tell me about Banque Misr payment cards" could
only ever come back `partial` — the page answered what was asked and was
rejected for not answering two things nobody asked.

**Narrowed, not deleted.** The `"compare"` trigger is kept: a comparison with
no stated dimension genuinely needs axes. Only the category-vocabulary half is
gone. This also keeps their `test_dynamic_plan_expansion` passing on its exact
expected sub-goal string.

### PATCH 3 — `validation/validator.py`: stopword list

Added generic question scaffolding (`offer`, `tell`, `about`, `show`, `where`,
`are`, `can`, …), generic classifiers (`type`, `types`, `kind`, `options`, …),
and this planner's own padding (`offerings`).

Measured: the words a *correct* page failed to match were almost entirely
scaffolding; the words a *wrong* page failed to match were the specific nouns
carrying the question (`cryptocurrency`, `martian`, `dog`, `student`,
`islamic`). Treating scaffolding as content made the topic test fire on the
wrong axis — penalising phrasing instead of subject matter.

### PATCH 4 — `validation/validator.py`: `_question_topics_present`

Lifts the topic test out of the generic fallback so the same rule can gate the
category branches. Runs against boilerplate-stripped body text.

> **Known limit, stated in the docstring:** the stripper locates the body by
> finding a breadcrumb, and the homepage has none — so on the homepage nothing
> is stripped and nav words *do* match. The homepage is handled by PATCH 5, not
> by this gate. Both are needed; neither is sufficient. Pinned by
> `test_the_gate_alone_does_not_save_the_homepage`.

### PATCH 5 — `validation/validator.py`: `overview` coverage — **the boilerplate false positive**

```python
if not target_fields or target_fields == ["overview"]:
    return (["overview"], [])          # never looked at the page
```

This was the only branch in `_evaluate_field_coverage` that ignored the
extraction. `overview` is the planner's default whenever a task names no
dimension, so for most questions coverage was declared complete before any
evidence was inspected — and the caller resolves on
`evidence and not missing_fields`, while *every* page yields some evidence.

That is the mechanism behind **the homepage resolving "how do I open an
account?"**: its nav tiles produced 6 evidence items and coverage waved them
through. Navigation would have stopped at hop 0, on the seed, and reported
success — collapsing the project's live-navigation claim to "fetched the
homepage, declared victory".

An overview now requires substance: entities, sections, or PDF tables.
Measured, content pages yield 6–12 entities each and the homepage yields 0.

A bare `tables` entry is deliberately **not** accepted: the site's currency
widget parses as a table on every page, headers `["{{fromCurrency}}", "Cash",
"Transfer"]`. Pages with real tables also carry entities, so nothing is lost.

### PATCH 6 — `validation/validator.py`: discovery branch

The branch resolved on `>= 3 matching entities **or** >= 5 entities of any
kind`, and the second half asks nothing about the question. Every content page
carries 6–12 entities, so any discovery question resolved on any content page.
Measured: *"what credit cards does Banque Misr offer?"* resolved on the
accounts-and-deposits page, reason `"Discovered 8 product entities on page
list."` Now gated on `_question_topics_present`; the entity counting is
untouched.

### PATCH 7 — `validation/validator.py`: category branches — **the rubber stamp**

The Accounts & Deposits, Consumer Loans and Fee-schedule branches each granted
`resolved=True` on a category-phrase match, with no test that the question's
subject was on the page. Measured, all three of these resolved against the
accounts hub with reason *"Found Accounts and Deposits offerings and details."*:

- *"Can I open a joint account with my dog?"*
- *"What is the interest rate on a Martian savings account?"*
- *"Does Banque Misr offer a cryptocurrency deposit account?"*

Any question containing a category word resolved against any page in that
category, so navigation stopped at the first hub it reached. One gate, computed
once, now guards all three. Their category conditions are untouched — membership
is still necessary, just no longer sufficient.

### PATCH 8 — `validation/validator.py`: `target_entity` inference

`sg_question.split(" for ")[-1]` kept the leading article and trailing
punctuation, so *"…the annual fee for the Classic credit card?"* produced the
entity `"the Classic credit card?"`, which matches no page — and
`_check_entity_relevance` then rejected the very page holding the answer. Also
switched to a single split so a question with two `" for "`s is not truncated.

### PATCH 9 — `validation/validator.py`: coverage never read entity names

Coverage searched table headers, sections, fields and `pdf_tables`, but not the
names of entities the extractor had just pulled off the page. A hub whose
answer *is* a set of documents scored zero: the fees hub carries entities like
*"Banque Misr payment cards Fees , Limits and commission"* and was still
reported as missing `fees`. Entity names now join the searchable text for every
branch. This can only add a place to find a field; it cannot mark one missing.

### PATCH 10 — `planning/planner.py`: `"document"` implied eligibility

`"document" in lower` added `eligibility` to the required fields, so *"show me
the fees and rates document"* demanded eligibility information and the page
holding the document would not resolve. Same shape as PATCH 2. Questions really
about eligibility still route via `"requirement"`.

### PATCH 11 — `validation/validator.py`: `interest_rate` coverage

The field is named `interest_rate` but the test looked only for `"installment"`
and `"tenor"`, so a page headed *"Deposits Interest Rates for individual
customers"* was reported as missing its interest rate. Added `interest`/`rate`.

### PATCH 12 — `validation/validator.py`: one topic rule, not two

The generic fallback carried its own inline *"at least half the query tokens
appear"* test, a second and different rule alongside the one now used by the
category branches — the setup where one branch accepts a page another rejects
for the same question. Both now call the same helper.

**This cost one false negative** (`"Show me the fees and rates document"`, which
the looser half-rule had let through: measured 0 FP / 1 FN with two rules
versus 0 FP / 2 FN with one). Unification was kept anyway: the extra failure is
on the same page as the other, so it costs one *phrasing*, not one
*destination*, and a permanently looser second rule is a standing false-positive
risk that 23 cases cannot rule out.

### PATCH 13 — tried and reverted

Adding extracted entity and section names to the topic gate's search text, on
the theory that a hub keeps much of its content in attachments rather than in
`focused_text`. It moved no case in the benchmark — "document" is genuinely
absent from the fees hub under any reading — so it was reverted rather than
shipped as untested surface area. The number is skipped so the remaining
numbering matches the marked blocks in the source.

### PATCH 14 — `verification/self_check.py`: unsupported prose survived

`finalize()` filtered *source URLs* to verified ones but only replaced the
answer text when **every** claim failed. With 3 of 4 verified, the unsupported
fourth stayed in the prose with verified sources printed beneath it — citation
laundering, which is precisely what the verification step exists to prevent.

`synthesize()` builds `draft_answer` out of claim statements verbatim, so a
failed claim can be located in the prose and struck. Now:

- every failed claim's sentence is removed, not just in the all-failed case;
- each removal is reported in `not_found` rather than vanishing silently;
- a bullet list stripped to nothing loses its orphaned lead-in;
- `metadata` carries `support_rate`, `claims_total`, `claims_supported`,
  `claims_unsupported`, `claims_contradicted` and `prose_removed` — all
  computed by `validate_answer` and previously dropped on the floor, leaving a
  caller no way to show how much of an answer was actually supported.

### PATCH 15 — `reasoning/synthesize.py`: a category noun in the answer

```python
statement = f"Banque Misr offers the following credit cards: {...}"
```

fired on *any* page yielding 3+ entities, so a loans run produced a final
answer reading *"Banque Misr offers the following credit cards: Personal
Loans, …"*. This was the only place the delivered answer named a product
category it had not established. Now neutral; `field` is `entity_list`.

---

## Not patched — reported instead

The following are Person B's to fix; they are recorded here so they are not
mistaken for accepted behaviour.

- **Topic coupling throughout.** `planner.py`, `validator.py` and
  `extractor.py` all branch on hardcoded product categories ("credit card",
  "loan", "account"/"deposit"/"certificate", "cards", "consumer loans", "fees
  and rates"). The patches above remove the cases that produced *wrong
  verdicts*, but the structure remains: delete the word "card" from their
  package and it stops working. This half of the project holds the opposite
  constraint.
- **`verification/self_check.py` hardcodes English superlatives** (`cheapest`,
  `best in the bank`) for its overbroad-claim check. Arabic and any other
  phrasing pass straight through.
- **Arabic.** Their extraction mangles Arabic text; every Arabic task validates
  as `unresolved`. That degrades in the safe direction — a visible failure, not
  a wrong answer — and they own it, so nothing here compensates for it.
- **`_check_entity_relevance` contains a product-specific special case**
  (`"classic" in norm_target and "accounts and deposits" in body_lower`).
  Harmless today, but it is category knowledge in a relevance test.
