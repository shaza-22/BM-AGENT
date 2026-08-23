"""Labelled benchmark for the vendored Person B validator.

Purpose
-------
The vendored validator decides when navigation stops. Tuning it by feel is how
you end up with a demo that either stops on the homepage or never stops at all,
so every change to it is measured here first.

Each case is ``(user_task, fixture_page, expected)`` where ``expected`` is
``resolve`` (this page genuinely answers the task) or ``reject`` (it does not).
Labels are grounded in what the fixture text actually contains -- checked with
grep, not assumed. Cases were written to describe the pages, *before* any fix
was chosen, so they are not tuned to a threshold.

The task is fed through Person B's own ``plan_task`` so the benchmark exercises
the same planner-rewritten question the live loop produces, padding and all.

Run: ``python3 scripts/validator_bench.py``
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from person_b.api import next_pending_sub_goal, plan_task, validate  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "live"

FEES = "home-pages-fees"
ACCOUNTS = "home-smes-retail-banking-accounts-and-deposits"
LOANS = "home-smes-retail-banking-consumer-loans"
CARDS = "home-smes-retail-banking-pages-cards"
CARD_LIST = "home-smes-retail-banking-pages-cards-credit-cards-list"
CLASSIC = "home-smes-retail-banking-pages-cards-credit-cards-pages-classic-credit-cards"
HOME = "index"

# (task, page, expected, note)
CASES: list[tuple[str, str, str, str]] = [
    # --- should resolve: the page really does answer the task -------------
    ("What credit cards does Banque Misr offer?", CARD_LIST, "resolve",
     "the credit-cards list page, 12 card entities"),
    ("What are the fees on the Classic credit card?", CLASSIC, "resolve",
     "classic card page carries 3 fee tables"),
    ("What is the annual fee for the Classic credit card?", CLASSIC, "resolve",
     "issuance/annual fee table present"),
    ("What personal loans are available?", LOANS, "resolve",
     "consumer loans page lists loan products"),
    ("What accounts and deposits does the bank offer?", ACCOUNTS, "resolve",
     "current/payroll/saving account entities present"),
    ("Where is the schedule of fees and commissions?", FEES, "resolve",
     "fees hub, 'Fees and Rates' + attachments section"),
    ("Show me the fees and rates document", FEES, "resolve",
     "same hub reached with different wording"),
    ("What is the daily ATM withdrawal limit?", CLASSIC, "resolve",
     "'withdrawal' x14 and 'limit' x15 on this page only"),
    ("Tell me about Banque Misr payment cards", CARDS, "resolve",
     "cards hub carries a descriptive body paragraph"),

    # --- should reject: right category, wrong subject ----------------------
    ("Does Banque Misr offer a cryptocurrency deposit account?", ACCOUNTS, "reject",
     "no crypto anywhere in the fixtures"),
    ("What is the interest rate on a Martian savings account?", ACCOUNTS, "reject",
     "no such product"),
    ("Can I open a joint account with my dog?", ACCOUNTS, "reject",
     "nonsense subject, in-category wording"),
    ("Does Banque Misr offer student accounts?", ACCOUNTS, "reject",
     "'student' appears in no fixture"),
    ("How do I open an Islamic account?", ACCOUNTS, "reject",
     "'Islamic' is nav-only; not on the accounts page"),
    ("What is the interest rate on a car loan?", LOANS, "reject",
     "loans page lists products, no car-loan rate"),

    # --- should reject: keyword present only in nav/footer boilerplate -----
    ("How do I open an account?", HOME, "reject",
     "homepage: account wording is nav chrome only"),
    ("What credit cards does Banque Misr offer?", HOME, "reject",
     "homepage nav mentions cards, lists none"),
    ("What personal loans are available?", HOME, "reject",
     "homepage nav only"),
    ("Where is the schedule of fees and commissions?", HOME, "reject",
     "homepage nav only"),
    ("Tell me about Islamic banking accounts", HOME, "reject",
     "'Islamic Banking' is a nav item; no content"),

    # --- should reject: wrong page entirely --------------------------------
    ("What personal loans are available?", CARDS, "reject", "loans task, cards page"),
    ("What are the fees on the Classic credit card?", LOANS, "reject", "cards task, loans page"),
    ("What credit cards does Banque Misr offer?", ACCOUNTS, "reject", "cards task, accounts page"),
]


def load_pages() -> dict[str, str]:
    return {
        p.stem: p.read_text(encoding="utf-8", errors="replace")
        for p in FIXTURES.glob("*.txt")
    }


def run() -> dict:
    pages = load_pages()
    fp: list[tuple] = []   # resolved but labelled reject
    fn: list[tuple] = []   # not resolved but labelled resolve
    ok = 0
    rows = []

    for task, page, expected, note in CASES:
        sub_goal = next_pending_sub_goal(plan_task(task))
        verdict = validate(
            sub_goal,
            pages[page],
            source_url=f"https://www.banquemisr.com/{page}",
        )
        got = "resolve" if verdict["resolved"] else "reject"
        status = verdict.get("status", "?")
        hit = got == expected
        ok += hit
        rows.append((hit, expected, got, status, task, page, verdict.get("reason", "")))
        if not hit:
            (fp if expected == "reject" else fn).append((task, page, status, note))

    return {
        "rows": rows,
        "total": len(CASES),
        "correct": ok,
        "false_positives": fp,
        "false_negatives": fn,
    }


def main() -> int:
    r = run()
    for hit, expected, got, status, task, page, reason in r["rows"]:
        mark = "ok " if hit else "XX "
        print(f"{mark}exp={expected:7} got={got:7} ({status:10}) | {task[:48]:48} | {page[:34]:34} | {reason[:44]}")
    print()
    print(f"correct         : {r['correct']}/{r['total']}")
    print(f"false positives : {len(r['false_positives'])}  (resolved a page that does not answer the task)")
    for t, p, s, note in r["false_positives"]:
        print(f"    FP  {t[:52]:52} on {p[:34]:34}  [{note}]")
    print(f"false negatives : {len(r['false_negatives'])}  (rejected a page that does answer it)")
    for t, p, s, note in r["false_negatives"]:
        print(f"    FN  {t[:52]:52} on {p[:34]:34}  ({s})  [{note}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def sweep() -> None:
    """Sweep the topic-gate tolerance and report FP/FN at each setting.

    The knob is the number of question content words allowed to be absent from
    the page. This is the measurement behind the value shipped in
    ``validator._TOPIC_MISS_TOLERANCE``; re-run it after any change to the
    stopword list, since the two interact.
    """
    from person_b.validation import validator

    original = validator._TOPIC_MISS_TOLERANCE
    print(f"{'tolerance':>10} {'correct':>9} {'FP':>4} {'FN':>4}   notes")
    print("-" * 78)
    try:
        for tol, label in [(0, "every content word must appear (shipped)"),
                           (1, "allow one word missing"),
                           (2, "allow two words missing"),
                           (99, "topic gate effectively off")]:
            validator._TOPIC_MISS_TOLERANCE = tol
            r = run()
            print(f"{tol:>10} {r['correct']:>7}/{r['total']:<2} {len(r['false_positives']):>4} "
                  f"{len(r['false_negatives']):>4}   {label}")
            for t, p, s, _n in r["false_positives"]:
                print(f"{'':>10}   + FP {t[:46]:46} on {p[:30]}")
            for t, p, s, _n in r["false_negatives"]:
                print(f"{'':>10}   - FN {t[:46]:46} on {p[:30]}")
    finally:
        validator._TOPIC_MISS_TOLERANCE = original

