# Vendored: Person B intelligence layer

This directory is a **vendored copy** of a package delivered by the second
author of the project (planning, extraction, validation, reasoning,
verification). It is not developed here. The goal is that a future drop from
them can be re-vendored mechanically, so every deviation from their zip is
recorded below.

## Source

| | |
|---|---|
| Archive | `banque_misr_person_b_blueprint_2.zip` |
| Package version | `0.1.0` (`src/Agent/pyproject.toml`) |
| Vendored on | 2026-08-23 |
| Their tests at vendor time | 70 passed |

## Re-vendoring procedure

1. Unzip the new drop.
2. `rm -rf src/person_b && cp -r <drop>/src/Agent src/person_b`
   (see *Rename* below — the drop may still ship the directory as `Agent`).
3. `rm -rf tests/person_b_vendor && cp -r <drop>/tests tests/person_b_vendor`,
   then re-apply the two test-harness adaptations below.
4. Re-apply the patches in `PATCHES.md` (each is a marked block; they are
   listed with the exact function and the reason).
5. `python3 -m pytest` — their suite and ours must both stay green.

---

## Deviations from the drop

### 1. Rename: `src/Agent/` -> `src/person_b/`

The zip ships the directory as `src/Agent/`, but **all 58 internal modules
import absolutely as `person_b.*`** (`from person_b.config import ...`), so the
package is unimportable as shipped. Their own docs claim it installs as-is.

Renaming the directory is the entire fix — **no import line was touched**. With
the rename alone their suite goes from "conftest ImportError" to 70 passed.

### 2. Their `pyproject.toml` / `requirements.txt` were not copied

`pdfplumber` and `python-dotenv` are already in this repo's `requirements.txt`,
and their `[build-system]`/`[tool.setuptools]` blocks would fight the root
`pyproject.toml`. Instead the root `pyproject.toml` gained `src` on the pytest
`pythonpath`, which is what makes `person_b` importable without packaging it.

### 3. Their `fixtures/live/` was not copied

Diffed against ours: their 17 files are an exact **English-only subset** of our
33, with an identical `manifest.json` schema. Copying theirs would only have
deleted the Arabic pages. Their tests now read this repo's `fixtures/live/`.

### 4. Test-harness adaptations (not bug fixes)

Their suite lives at `tests/person_b_vendor/`. Two changes were needed:

- **`conftest.py`: dropped the `sys.path.insert` for `src/`** and repointed
  `fixtures_dir` from `<pkg>/fixtures/live` to the repo's `fixtures/live`.
  Both of their paths assumed the package was its own repo root. `pythonpath`
  in `pyproject.toml` now does the sys.path job, so only one mechanism is in
  play rather than two that can disagree.
- **Directory named `person_b_vendor`, not `person_b`.** Their tests carry
  `__init__.py` files, so pytest names the package after the directory. Called
  `tests/person_b/`, that package **shadowed the real `person_b` package** and
  their own conftest failed to import. The suffix is the whole fix; their
  `__init__.py` files are kept exactly as shipped.

### 5. `examples/person_a_integration_smoke.py` — written, not stricken

Referenced by their `README.md:599`, `docs/PERSON_B_HANDOFF.md:38` and `:118`,
and `docs/tasks/TASK_04_...md:260` — **but absent from the zip**. Their handoff
points at it as the first thing to run.

It is written here at the referenced path rather than striking the references,
so their docs stop lying and a one-command offline check exists. It drives the
full contract — `plan_task -> validate -> apply_validation -> expand_plan ->
synthesize -> validate_answer -> finalize` — over `fixtures/live/`, with no
network and no model, and exits non-zero if any stage breaks shape:

    python examples/person_a_integration_smoke.py

Their docs are vendored at `docs/person_b/`. A future drop that ships its own
`examples/` should replace this file.

### 6. Behavioural patches

Three validator/extraction bugs and one verification bug were found by probing
their code against this repo's fixtures. They could not be turned around
upstream in time, so they are patched here. Every patch is a marked block
(`# --- PATCH n (vendor) ... # --- END PATCH n ---`) and is documented, with
before/after measurements, in **`PATCHES.md`**.
