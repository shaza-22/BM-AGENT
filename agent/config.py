"""
Navigation policy for the Banque Misr research agent.

What it does
    Holds the agent's tunables: where a run starts, how far it may go, which
    model picks the links, and how candidate links are scored before they reach
    that model.

Inputs
    None. Pure data.

Outputs
    Module-level constants.

Why it is needed
    Separates *policy* (how boldly to navigate, what to spend) from
    *perception* (``browsing/config.py``, how to read a page). It also isolates
    the one hardcoded URL the project is allowed to contain -- the seed. Every
    other destination must be discovered by following links at runtime.

Topic-agnosticism
    Nothing here names a product, category or page. The ranking weights below
    read only structural facts about a link (where on the page it was found,
    when it was discovered, how often it has been offered). None of them looks
    at the sub-goal text, which is what keeps a lexical filter from quietly
    becoming the agent.
"""

from __future__ import annotations

# --- Where a run starts ----------------------------------------------------
# The only URL in the source tree. Everything else is discovered live.
SEED_URL = "https://www.banquemisr.com/"

# --- Language --------------------------------------------------------------
# Whether a run may follow links in the other language.
#
#   "penalise" -- other-language links stay in the frontier, ranked below the
#                 task's own. Chosen because the site is densely interlinked
#                 (a language switcher sits on every page) and because some
#                 content exists in one language only. Strict would report
#                 "not found on the website" for content that is right there
#                 in the other language -- a false negative, which is the
#                 worst possible answer for missing-information handling.
#   "strict"   -- other-language links are dropped in extract_links.
CROSS_LANGUAGE_POLICY = "penalise"

# Applied to a link whose language differs from the run's. Big enough to keep
# same-language links on top, small enough that the only route to an answer is
# still reachable. Sits with the other structural weights below.
PENALTY_OTHER_LANGUAGE = 1.0


def seed_for(language: str | None) -> str:
    """The starting URL for a run in *language*.

    Derived from the one hardcoded URL rather than adding a second: the Arabic
    site is the same paths carrying "?sc_lang=ar-EG", so the seed is the
    homepage plus that marker.
    """
    from browsing import config as browsing_config

    if not language or language == browsing_config.SITE_DEFAULT_LANGUAGE:
        return SEED_URL
    tag = browsing_config.LANGUAGE_URL_TAGS.get(language)
    if not tag:
        return SEED_URL
    joiner = "&" if "?" in SEED_URL else "?"
    return f"{SEED_URL}{joiner}{browsing_config.LANG_PARAM}={tag}"


# --- Caps ------------------------------------------------------------------
# Observed depth: homepage -> category -> list -> product detail is 3 hops, and
# a linked fee PDF adds a 4th, so 5 leaves one spare for a wrong turn.
MAX_HOPS = 5
MAX_PAGES = 15
# Measured on the real homepage: 50 candidates is ~5.6k characters, roughly
# 1,400-1,650 prompt tokens. See README for the per-hop cost.
CANDIDATE_LIMIT = 50

# --- Model -----------------------------------------------------------------
# Which LLMClient implementation agent.llm.make_llm_client() builds.
PROVIDER = "gemini"  # "gemini" | "claude"

# Both providers read their key from the environment, and agent.llm also loads
# a .env file at the project root. Only the *names* of these variables appear
# anywhere in the code or the logs -- never their values.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Gemini. The free tier's usual workhorse; override if your quota differs.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_MAX_TOKENS = 4096

# Gemini reasons before answering by default, and link selection does not need
# it: the model is picking one entry from a list of labels, not solving
# anything. Measured on a two-hop run, same prompt size both times: 3.5s for the
# first call against 32.8s for the second, with 35.6s of a 38.7s run spent
# inside the model. Set to None to let the model decide.
#
# Two knobs because the API changed: older models take a token budget (0 turns
# reasoning off), newer ones take a level. Whichever is set is sent; if the API
# rejects it, the client logs a warning, drops it and carries on, so a model
# that supports neither still works.

# gemini-3.x rejects thinking_budget outright (a bare 400), so the level is the
# working knob there; the budget is kept for older models. Measured on 3.6-flash:
# thinking_level="low" took a 32.8s call down to 1.6s with the choice quality and
# the arrival signal both intact.
GEMINI_THINKING_BUDGET: int | None = None
GEMINI_THINKING_LEVEL: str | None = "low"

# Claude. Link selection is a judgement call over a short list, not a research
# task, so low effort keeps adaptive thinking brief. Note: do NOT disable
# thinking on this model to save tokens -- with thinking off it sometimes writes
# a tool call into visible text and can leak reasoning tags. Lower effort instead.
CLAUDE_MODEL = "claude-opus-5"
CLAUDE_EFFORT = "low"
CLAUDE_MAX_TOKENS = 4096
# Self-reported LLM confidence is weakly calibrated, so nothing branches on it
# by default. Raise this to make the navigator treat low-confidence picks as
# "no candidate".
MIN_CONFIDENCE = 0.0

# --- LLM transport resilience ----------------------------------------------
# Free-tier endpoints return 503 UNAVAILABLE regularly, and a single transient
# failure used to end a whole navigation run. The selector is called once per
# hop, so one retried call costs seconds where a lost run costs the task.
LLM_MAX_ATTEMPTS = 4          # total attempts, i.e. 3 retries
LLM_RETRY_BACKOFF_S = 2.0     # doubled each attempt
LLM_RETRY_MAX_BACKOFF_S = 30.0

# Worth retrying: overload, rate limiting, gateway and timeout failures.
LLM_TRANSIENT_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# Settled answers -- a bad key or an unknown model will not fix itself, and
# retrying only delays a clear error message.
LLM_PERMANENT_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 405, 422})
# Google returns these alongside the HTTP code; some transports surface only
# the name, so both are checked.
LLM_TRANSIENT_STATUS_NAMES: frozenset[str] = frozenset(
    {"UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE_EXCEEDED", "ABORTED"}
)

# When a request is refused with a 400 that names no field, optional settings
# are dropped one at a time in this order and the call retried, most-likely
# culprit first. The schema comes last because losing it costs the most.
LLM_OPTIONAL_SETTING_DROP_ORDER: tuple[str, ...] = ("thinking", "schema")

# A quota whose id matches one of these refills on a daily schedule, so no
# amount of backing off inside one run will clear it.
LLM_DAILY_QUOTA_MARKERS: tuple[str, ...] = ("perday", "per_day", "daily")

# --- Session memory --------------------------------------------------------
SESSION_MAX_TURNS = 8          # older turns drop out of the resolver's context
SESSION_TTL_S = 3600           # sessions are in-memory only; this bounds the leak
SESSION_MAX_SESSIONS = 200     # LRU cap, so a long-running server cannot grow forever
MAX_TASK_CHARS = 2000          # a task string goes straight into a prompt
RESOLVER_MAX_SUB_GOAL_CHARS = 400  # longer than this means the model answered, not rewrote

# Score bonus for a link whose page was visited in an earlier turn of the same
# session. SHIPPED DISABLED -- see the taxonomy in agent/session.py for why.
# Enabling it is safe (it only reorders links the agent discovered this run),
# but it complicates the project's central claim for a small saving.
FOLLOW_UP_PATH_BONUS = 0.0

# --- Candidate ranking weights ---------------------------------------------
# Additive score; higher is offered sooner. All signals are structural.
WEIGHT_CURRENT_PAGE = 3.0        # found on the page we are standing on
WEIGHT_RECENCY = 1.0             # divided by (1 + hops since discovery)
WEIGHT_SOURCE = {"body": 0.5, "footer": 0.25, "nav": 0.0}

# The nav is identical on every page (60 of 173 observed links appear on all of
# them), so after the first hop it carries almost no page-specific information.
# It is penalised rather than dropped because it is also the only route into a
# different section when an earlier hop went down the wrong branch.
PENALTY_NAV_AFTER_FIRST_HOP = 0.75

# A link the model saw and did not choose is demoted, never removed: it was not
# rejected, merely not ranked first, and it is exactly what should be tried
# when the first choice dead-ends.
PENALTY_PER_OFFER = 0.5

# See browsing.extract_links.alias_key -- the same page under the other URL
# scheme. Deprioritised and logged, never dropped, because the equivalence is
# a heuristic.
PENALTY_ALIAS_VISITED = 1.5

# No page region may be crowded out of the offered list entirely.
#
# Measured on the real homepage: 40 body links outscore 11 footer ones, so a
# straight top-40 cut offered zero footer links -- putting the footer-only fees
# hub and the sitemap out of reach on the first hop, and with them a whole
# class of sub-goal. The footer runs 11-13 links and is nearly identical on
# every page, so reserving enough to cover it entirely is cheap and removes any
# dependence on where in the footer a given link happens to sit.
#
# This is a structural diversity guarantee. It reserves room for a *region*,
# never for a particular destination.
REGION_RESERVED_SLOTS = {"footer": 13, "nav": 6}

# --- WAF detection ---------------------------------------------------------
# The site sits behind an F5 BIG-IP WAF that answers HTTP 200 with a block
# page, so fetch_page reports ok=True and the agent would happily parse the
# refusal as content and cite it. Detected as: a marker below AND almost no
# links (a real page carries 70+). Hitting a WAF that has already flagged you
# is how a run turns into a ban, so detection aborts the whole navigation.
WAF_BLOCK_MARKERS: tuple[str, ...] = (
    "the requested url was rejected",
    "your support id is",
    "access denied",
)
WAF_MAX_LINKS = 5


# --- Multi-sub-goal loop budgets -------------------------------------------
# Caps for agent/loop.py, which drives Person B's plan across several
# navigations. These override PersonBConfig at construction time; their
# defaults are deliberately left untouched in src/person_b/config.py so a
# future drop from them re-vendors cleanly (see src/person_b/VENDORED.md).
#
# Their shipped default is max_expansion_sub_goals = 20. That is unusable
# here. Measured on the fixtures, their extractor pulls 12 entities off the
# credit-cards list page, and expansion turns each into its own sub-goal: 12
# fresh navigations of up to MAX_PAGES pages each is roughly 180 requests at a
# site behind an F5 WAF that bans on burst traffic, and 12+ model calls
# against a free tier that allows 20 per day. The plan would exhaust the
# quota before the first answer.
#
# The per-sub-goal caps above (MAX_HOPS, MAX_PAGES) do not bound this on their
# own -- they multiply. The two budgets that actually hold the line are the
# global ones, which are spent across the whole plan rather than reset per
# sub-goal.
MAX_SUB_GOALS = 4  # 1 initial + 3 expansions
MAX_EXPANSION_DEPTH = 1  # an expanded sub-goal never expands again
LOOP_MAX_PAGES = 25  # GLOBAL across all sub-goals, not per sub-goal
LOOP_MAX_LLM_CALLS = 12  # hard stop well inside a 20/day free tier

# A partial verdict does not stop navigation. "Partial" means the entity is
# relevant and some requested fields were found -- precisely the state where
# the rest is one hop deeper, on a detail page or a linked PDF. Stopping there
# throws away the remaining hop budget; discarding the evidence throws away
# work already paid for. So the loop keeps navigating and keeps the partial as
# a fallback: if nothing resolves, the answer is synthesised from accumulated
# partials with the gaps named, which beats reporting a flat failure.
PARTIAL_STOPS_NAVIGATION = False


# --- Acceptance gate -------------------------------------------------------
# A second, independent check on the vendored validator's ``resolved``, run
# from a vantage point it does not have: the whole run rather than one page.
# See agent/acceptance.py for the argument, and src/person_b/PATCHES.md for
# the validator bugs it sits alongside (it is not a substitute for those --
# both are needed, and the gate abstains on the first pages of a run).
#
# It is one-way: it can withhold a resolve, never grant one.
ACCEPTANCE_GATE_ENABLED = True

# How many *other* pages this run must also carry a snippet before it counts
# as boilerplate rather than content. 2 rather than 1 because two pages in a
# section can legitimately share a sentence; a third occurrence is chrome.
GATE_MIN_REPEATS = 2

# Below this length a snippet is not distinctive enough to conclude anything
# from -- "Cash" appears on every page of a bank site and proves nothing.
GATE_MIN_SNIPPET_CHARS = 24


# --- Answer composition ----------------------------------------------------
# The vendored synthesis is template-based with no model in it, so it can fill
# slots but cannot write prose. agent/answer.py asks the model for the final
# wording, constrained to the claims the run verified, and agent/grounding.py
# strikes any sentence the evidence does not support before it is shown.
#
# Ships on. Costs one model call per task, reserved out of LOOP_MAX_LLM_CALLS
# below so navigation cannot spend the whole budget and leave nothing to write
# the answer with.
#
# The template answer is always produced first and is what stands whenever
# composition is skipped, fails, or is struck empty -- so this can only ever
# improve the wording, never be the reason a run has no answer.
COMPOSE_ANSWER = True

# How many verified facts to put in front of the model. A page can yield 35;
# the prompt stays small and the grounding set stays the same set the model
# was shown, which is what makes "every figure must be quoted" checkable.
COMPOSE_MAX_FACTS = 40


# --- Agentic planning ------------------------------------------------------
# The vendored planner decomposes by keyword and, measured across eight varied
# tasks, produced exactly one sub-goal every time -- a plan panel showing one
# restated line. agent/planner.py asks the model to decompose instead, and
# falls back to the keyword planner on any failure.
#
# Set False to revert to deterministic planning in one line. Everything
# downstream is unchanged either way: the sub-goals are navigated, validated,
# expanded and reported by exactly the same code.
LLM_PLANNING = True

# Cap on what the planner may return, before any expansion.
#
# The model is called once per hop, measured. With LOOP_MAX_LLM_CALLS = 12 and
# one call reserved for composing the answer, planning takes one more and
# leaves ten for navigation. Pages here resolve in one to three hops, so:
#
#     3 sub-goals x 3 hops = 9, plus plan and compose = 11   fits
#     4 sub-goals x 3 hops = 12, plus plan and compose = 14  does not
#
# A sub-goal that runs to MAX_HOPS spends the budget early; the remainder are
# then marked not-available with the reason and appear in the answer, so the
# overrun is visible rather than silent.
MAX_PLANNED_SUB_GOALS = 3


# --- LLM extraction fallback -----------------------------------------------
# The vendored extraction is keyword matching. When it yields nothing usable,
# agent/extraction.py reads the page with the model instead -- one call per
# sub-goal, at most, and only after the deterministic path has failed.
#
# The label the model writes is free (that is how "Fees and Rates" on a page
# bridges to "fees and charges" in a question); the value must be a literal
# substring of the page. See that module for why.
#
# Re-enabled after the two bugs it was switched off for were root-caused and
# fixed: the multi-table repetition (a sentence-length table name repeated in
# front of every row, and the label column chosen by position rather than by
# content) and template placeholders passing the verbatim check. Both are
# PATCH 19 in src/person_b/PATCHES.md, and both are pinned by tests.
#
# Verified against three page shapes -- a fee table, a list of names whose
# marker the deterministic extractor needs, and a multi-table limits page --
# in tests/test_extraction.py::TestThreePageShapes.
#
# Set False to revert to deterministic extraction only.
LLM_EXTRACTION_FALLBACK = True

# How much of a page to show the model. Long enough for a fee table, short
# enough that a 140k-character PDF does not become the prompt.
EXTRACTION_MAX_PAGE_CHARS = 12000

# Facts to accept from one page. A page has many numbers on it; this is a
# ceiling on the answer's size, not a target.
EXTRACTION_MAX_FACTS = 25


# --- Tiered answers --------------------------------------------------------
# The spec asks the agent to "analyze, compare, or summarize ... when
# required". Analysis means derived statements -- "issuance and renewal are
# both EGP 250, so there is no increase after year one" is on no page, though
# both figures are. Composed prose alone produced only literal restatements,
# because the prompt forbade anything else.
#
# Three kinds of sentence, each with a different warranty:
#   fact      restated from a page
#   analysis  derived by comparison, aggregation or summary
#   judgment  a recommendation, only where the task asks to be advised
#
# Costs nothing extra: it is the shape the existing composition call's reply
# takes, not another call. agent/grounding.py runs on every sentence
# unchanged -- an invented figure dies in tier 2 exactly as it does in tier 1.
# Only the inference is the agent's, never the figures.
#
# Set False to go back to untiered prose (which itself falls back to the
# template answer).
ANSWER_TIERS = True
