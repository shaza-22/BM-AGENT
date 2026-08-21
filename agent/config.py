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
