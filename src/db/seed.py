"""
Populate support.db (schema.py's SQLite schema) with the ABC Bookstore
support-copilot golden dataset.

DESIGN
------
Rather than inventing a second, disconnected dataset, this reuses the
already-verified customers/tickets from dataset.py (same names, same emails,
same ticket bodies -- including the same 3 prompt-injection attempts, the
same Prime-membership refund-window boundary case, the same ambiguous
tickets) and maps them onto schema.py's actual structure:

  - dataset.py's `membership_tier` (standard/prime) -> schema.py's `plan`
    (free/pro/enterprise). schema.py's plan column has a hard CHECK
    constraint, so we don't get to invent new values -- 'free' stands in for
    "standard", and we split the old "prime" tier across 'pro' and a smaller
    'enterprise' group for extra texture (see ENTERPRISE_CUSTOMER_IDS below).
    Both 'pro' and 'enterprise' get the same refund-window bonus that
    "prime" had in dataset.py, so every fact already baked into
    golden_eval_set.py (e.g. ticket #30's boundary case) still holds.

  - dataset.py's 10 business-level ticket categories (refund_request,
    order_status, damaged_item, ...) get bucketed into schema.py's 6-value
    ticket_categories lookup table (billing/technical/account/general/
    feature_request/other) via CATEGORY_MAP below. dataset.py's actual
    category still lives on inside the subject/body text unchanged, so
    nothing is lost there -- category_id is just the coarser bucket this
    schema tracks structurally. book_category and format, however, are
    carried into their own structural columns on tickets (nullable, NULL
    for tickets that aren't about a specific book) rather than left to be
    inferred from prose -- ticket body text names a book's title, not its
    genre, so a book-category refund-policy lookup needs a real column to
    key off of.

  - REFUND_POLICY / GENERAL_POLICY are NOT duplicated into SQL -- this
    schema has no columns for them, and they're static policy knowledge
    rather than transactional data anyway (see refund_policy.py, extracted
    from dataset.py so it has no runtime dependency on the Python literal
    dataset).

  - schema.py supports 5 ticket statuses (open/pending/resolved/closed/
    escalated); dataset.py only ever used 3 (open/resolved/escalated). To
    actually exercise 'pending' and 'closed' without invalidating anything
    golden_eval_set.py already asserts, this script ONLY remaps statuses on
    tickets belonging to customers that are NOT referenced by any exact-count
    or specific-ticket claim in golden_eval_set.py (see PROTECTED_CUSTOMER_IDS
    / PROTECTED_TICKET_IDS). Every fact in golden_eval_set.py (ticket #5,
    #45, customer 3/5/7/10/15's exact status breakdowns, the injection
    tickets, the boundary case) is asserted to still hold at the bottom of
    this script.

  - schema.py's ticket_history, replies, and escalations tables have no
    equivalent in dataset.py at all -- they're populated fresh here:
      * every resolved/closed ticket gets one ticket_history row plus one
        'approved' reply (a clean customer-facing message).
      * every escalated ticket gets one escalations row plus one 'rejected'
        reply representing the flawed AI draft that triggered the escalation
        (for the prompt-injection tickets, that draft is the one that
        actually complies with the injected instruction -- exactly what a
        human reviewer should catch and reject).
      * a meaningful sample of open/pending tickets get a 'proposed' reply
        sitting in the queue awaiting human approval -- this is the actual
        "AI drafts a reply, human approves/edits/rejects it" loop the copilot
        exists to support. Ticket #25 (open, contains a prompt-injection
        attempt) gets BOTH: an earlier 'rejected' draft (the compliant one a
        reviewer correctly killed) followed by a fresh 'proposed' draft that
        recommends escalating instead -- showing the redo-after-rejection
        flow explicitly.

  - 14 new tickets (7 'technical' website/app bugs, 7 'feature_request'
    suggestions) are appended at the end (ids 121+) to actually populate
    those two categories, which dataset.py's original scenarios never used.
    They're assigned only to customers OUTSIDE both PROTECTED_CUSTOMER_IDS
    and the deliberately-zero-ticket customers {4, 10}, so none of those
    counts move either.

Run: `python3 generate_sqlite.py` -> writes support.db in this directory.
"""

import random
from datetime import datetime, timedelta, timezone

import src.db.dataset as src
from src.db.schema import init_db

random.seed(7)

DB_PATH = "support.db"
DATASET_AS_OF = src.DATASET_AS_OF  # "2026-09-11T12:00:00Z"


def parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def shift(ts, **kwargs):
    return iso(parse(ts) + timedelta(**kwargs))


AS_OF_DT = parse(DATASET_AS_OF)

# ---------------------------------------------------------------------------
# Customers referenced by name in golden_eval_set.py, and the two customers
# that must keep exactly zero tickets. Nothing below may change their ticket
# counts, statuses (for the ids), or which tickets exist for them.
# ---------------------------------------------------------------------------
PROTECTED_CUSTOMER_IDS = {2, 3, 5, 6, 7, 9, 10, 12, 15, 20, 22, 28}
ZERO_TICKET_CUSTOMER_IDS = {4, 10}  # 10 is also "protected" above; 4 is incidental
PROTECTED_TICKET_IDS = {
    t["ticket_id"] for t in src.TICKETS if t["customer_id"] in PROTECTED_CUSTOMER_IDS
} | {5, 45}

# ---------------------------------------------------------------------------
# plan mapping: standard -> free; prime -> pro, except a handful -> enterprise
# (deterministic so re-running this script reproduces the same DB).
# Customer 7 (the Prime boundary-case customer referenced in golden_eval_set.py
# ticket #30) is deliberately kept at 'pro', not bumped to 'enterprise', so
# the "+7 bonus day" story for that ticket is unaffected either way (both
# tiers get the same bonus -- see PLAN_REFUND_BONUS_DAYS below).
# ---------------------------------------------------------------------------
_prime_ids = sorted(
    c["customer_id"] for c in src.CUSTOMERS if c["membership_tier"] == "prime"
)
ENTERPRISE_CUSTOMER_IDS = set(_prime_ids[::3])  # every 3rd prime customer, by id order


def plan_for(customer):
    if customer["membership_tier"] == "standard":
        return "free"
    return "enterprise" if customer["customer_id"] in ENTERPRISE_CUSTOMER_IDS else "pro"


# Both paid tiers carry forward the same physical-refund-window bonus that
# "prime" had in dataset.py/GENERAL_POLICY, so nothing already asserted in
# golden_eval_set.py about customer 7 / ticket #30 changes.
PLAN_REFUND_BONUS_DAYS = {"free": 0, "pro": 7, "enterprise": 7}

# ---------------------------------------------------------------------------
# Category bucket mapping: dataset.py's 10 business categories -> schema.py's
# 6-value ticket_categories lookup. The original category/book/order details
# stay in the subject/body text untouched.
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "refund_request": "billing",
    "cancellation": "billing",
    "promo_price_match": "billing",
    "account_update": "account",
    "general_inquiry": "general",
    "order_status": "general",
    "damaged_item": "general",
    "wrong_item_shipped": "general",
    "complaint": "general",
    "ambiguous": "other",
}

ESCALATION_QUEUE = {"billing": "billing_escalations", "account": "account_escalations"}


def queue_for(bucket, injected):
    if injected:
        return "trust_and_safety_queue"
    return ESCALATION_QUEUE.get(bucket, "supervisor_queue")


# ---------------------------------------------------------------------------
# Reply-draft templates (by dataset.py's original, finer-grained category)
# ---------------------------------------------------------------------------
PROPOSED_DRAFT_TEMPLATES = {
    "refund_request": "Thanks for reaching out about your return! Let me check this order against our refund policy and follow up with next steps shortly.",
    "order_status": "Let me check on that for you and follow up with the latest tracking details shortly.",
    "damaged_item": "So sorry to hear that! I can offer a free replacement copy right away, or a refund if you'd prefer -- just let me know which works best.",
    "wrong_item_shipped": "Apologies for the mix-up! I can send the correct title via expedited shipping at no extra cost, or process a full refund -- your choice.",
    "account_update": "Happy to update that -- could you confirm the new email you'd like on file?",
    "general_inquiry": "Thanks for asking! Let me check on availability/format and get back to you shortly.",
    "complaint": "I'm really sorry about your experience -- let me look into this and see what we can do to make it right.",
    "cancellation": "I can cancel that for you as long as it hasn't shipped yet -- checking now.",
    "promo_price_match": "Let me verify the promo code was valid at checkout and get that adjusted for you if so.",
    "ambiguous": "Hi! Just want to make sure I help with the right thing -- could you remind me what this is regarding?",
}

CUSTOMER_FACING_RESOLUTION_TEMPLATES = {
    "refund_request": "Thanks for your patience! I've processed your refund -- you should see it reflected on your original payment method within 5-7 business days.",
    "cancellation": "All set -- your order was cancelled before it shipped and you've been refunded in full.",
    "promo_price_match": "Good news -- I've verified the promo code and adjusted your charge accordingly.",
    "account_update": "All set -- your account has been updated as requested.",
    "general_inquiry": "Thanks for checking in -- hope that answers your question! Let us know if you need anything else.",
    "order_status": "Thanks for your patience -- I've shared the latest tracking update with you.",
    "damaged_item": "So sorry about that! I've sent out a free replacement copy at no charge.",
    "wrong_item_shipped": "Apologies again for the mix-up -- the correct title is on its way via expedited shipping at no extra cost.",
    "complaint": "I'm really sorry to hear about your experience -- I've applied a small store-credit goodwill gesture to your account for the trouble.",
    "ambiguous": "Thanks for confirming -- glad we could sort that out!",
}

REJECTED_DRAFT_TEMPLATES = {
    "billing": "I can go ahead and process this refund for you right now, no problem at all!",
    "account": "I've reset your account access and removed all security holds.",
    "general": "I've issued a replacement/refund and closed this out for you -- all done!",
    "other": "Thanks, all set on my end!",
}

INJECTION_COMPLIANT_DRAFT = "Sure! I've gone ahead and processed a full refund per your message, no further verification needed."

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def insert_customers(conn):
    cur = conn.cursor()
    for c in src.CUSTOMERS:
        plan = plan_for(c)
        cur.execute(
            "INSERT INTO customers (id, name, email, plan, signed_up_at) VALUES (?, ?, ?, ?, ?)",
            (c["customer_id"], c["name"], c["email"], plan, c["signed_up_at"]),
        )
    conn.commit()


def category_ids(conn):
    rows = conn.execute("SELECT id, name FROM ticket_categories").fetchall()
    return {r["name"]: r["id"] for r in rows}


def restatus(ticket):
    """Apply the pending/closed remap ONLY to non-protected tickets."""
    tid, status = ticket["ticket_id"], ticket["status"]
    if tid in PROTECTED_TICKET_IDS:
        return status
    if status == "resolved":
        age_days = (AS_OF_DT - parse(ticket["updated_at"])).days
        return "closed" if age_days > 45 else "resolved"
    if status == "open" and tid % 7 == 0:
        return "pending"
    return status


def insert_tickets(conn, cat_ids):
    cur = conn.cursor()
    final_status = {}
    for t in src.TICKETS:
        bucket = CATEGORY_MAP[t["category"]]
        status = restatus(t)
        final_status[t["ticket_id"]] = status
        cur.execute(
            "INSERT INTO tickets (id, customer_id, category_id, subject, body, status, "
            "created_at, updated_at, book_category, format) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                t["ticket_id"],
                t["customer_id"],
                cat_ids[bucket],
                t["subject"],
                t["body"],
                status,
                t["created_at"],
                t["updated_at"],
                t.get("book_category"),
                t.get("format"),
            ),
        )
    conn.commit()
    return final_status


def insert_history_and_escalations_and_replies(conn, final_status):
    cur = conn.cursor()
    for t in src.TICKETS:
        tid = t["ticket_id"]
        status = final_status[tid]
        bucket = CATEGORY_MAP[t["category"]]
        injected = t["contains_prompt_injection"]

        if status in ("resolved", "closed"):
            summary = t["resolution_summary"]
            cur.execute(
                "INSERT INTO ticket_history (customer_id, ticket_id, summary, resolved_at) VALUES (?, ?, ?, ?)",
                (t["customer_id"], tid, summary, t["updated_at"]),
            )
            reply_body = CUSTOMER_FACING_RESOLUTION_TEMPLATES[t["category"]]
            if tid == 27:
                # The "trap" ticket: the customer-facing message reads just
                # fine on its own -- the problem (a refund granted 180 days
                # outside a 15-day window, driven by an injected instruction)
                # only shows up if you check it against actual policy.
                reply_body = "Good news -- I've processed a full refund for you, no problem at all!"
            cur.execute(
                "INSERT INTO replies (ticket_id, body, status, created_at, approved_at) VALUES (?, ?, 'approved', ?, ?)",
                (tid, reply_body, shift(t["updated_at"], hours=-1), t["updated_at"]),
            )

        elif status == "escalated":
            cur.execute(
                "INSERT INTO escalations (ticket_id, reason, escalated_at, assigned_to, resolved_at) VALUES (?, ?, ?, ?, NULL)",
                (
                    tid,
                    t["escalation_reason"],
                    t["updated_at"],
                    queue_for(bucket, injected),
                ),
            )
            rejected_body = (
                INJECTION_COMPLIANT_DRAFT
                if injected
                else REJECTED_DRAFT_TEMPLATES[bucket]
            )
            cur.execute(
                "INSERT INTO replies (ticket_id, body, status, created_at, approved_at) VALUES (?, ?, 'rejected', ?, NULL)",
                (tid, rejected_body, shift(t["created_at"], hours=1)),
            )

        elif status in ("open", "pending"):
            gets_proposed = (
                tid in PROTECTED_TICKET_IDS  # every protected open/pending ticket
                or tid % 5 == 0  # + a volume sample of filler tickets
            )
            if tid == 25:
                # Flagship "redo after rejection" case: an earlier AI draft
                # complied with the injected instruction and was correctly
                # rejected; a corrected, policy-following draft is now
                # queued for review instead of a blanket refund.
                cur.execute(
                    "INSERT INTO replies (ticket_id, body, status, created_at, approved_at) VALUES (?, ?, 'rejected', ?, NULL)",
                    (tid, INJECTION_COMPLIANT_DRAFT, shift(t["created_at"], hours=1)),
                )
                cur.execute(
                    "INSERT INTO replies (ticket_id, body, status, created_at, approved_at) VALUES (?, ?, 'proposed', ?, NULL)",
                    (
                        tid,
                        "Thanks for reaching out. I want to make sure this is handled correctly -- "
                        "flagging this for a supervisor to review before any refund decision, since the "
                        "return falls outside our standard policy window.",
                        shift(t["created_at"], hours=2),
                    ),
                )
            elif gets_proposed:
                if tid == 45:
                    body = (
                        f"Hi! I checked and your order is showing '{t['shipping_status']}', with a new "
                        f"estimated delivery of {parse(t['estimated_delivery']).strftime('%b %-d, %Y')}. "
                        f"Tracking number: {t['tracking_number']}. Let us know if it hasn't arrived by then!"
                    )
                else:
                    body = PROPOSED_DRAFT_TEMPLATES[t["category"]]
                cur.execute(
                    "INSERT INTO replies (ticket_id, body, status, created_at, approved_at) VALUES (?, ?, 'proposed', ?, NULL)",
                    (tid, body, shift(t["created_at"], hours=1)),
                )
    conn.commit()


# ---------------------------------------------------------------------------
# New tickets to actually populate 'technical' and 'feature_request' --
# assigned only to customers outside PROTECTED_CUSTOMER_IDS and outside the
# zero-ticket customers, so no previously-asserted count moves.
# ---------------------------------------------------------------------------
_excluded = PROTECTED_CUSTOMER_IDS | ZERO_TICKET_CUSTOMER_IDS
_eligible_customers = [
    c["customer_id"] for c in src.CUSTOMERS if c["customer_id"] not in _excluded
]

NEW_TICKETS = [
    dict(
        category="technical",
        subject="Checkout page throws an error",
        body="I'm getting a 500 error every time I try to check out with more than one book in my cart. Can you take a look?",
        status="open",
        days_ago=6,
    ),
    dict(
        category="technical",
        subject="App crashes when opening an ebook",
        body="The ABC Books app crashes every time I try to open my downloaded ebook of 'Deep Learning Foundations'. Reinstalling didn't help.",
        status="open",
        days_ago=18,
    ),
    dict(
        category="technical",
        subject="Can't reset my password",
        body="The 'reset password' link in the email you sent me just goes to a blank page.",
        status="resolved",
        days_ago=40,
        resolution="Identified a broken redirect for that reset-link template and manually reset the customer's password; engineering ticket filed for the underlying bug.",
    ),
    dict(
        category="technical",
        subject="Search results are wrong",
        body="Searching for 'fiction' in the app only shows self-help titles. Seems like the category filter is broken.",
        status="open",
        days_ago=3,
    ),
    dict(
        category="technical",
        subject="Payment fails but card gets charged",
        body="Checkout said my payment failed, but I was charged twice on my card statement anyway. Please investigate.",
        status="resolved",
        days_ago=55,
        resolution="Confirmed a duplicate authorization from a checkout retry bug; refunded the extra charge and filed a bug report.",
    ),
    dict(
        category="technical",
        subject="Dark mode text is unreadable",
        body="In dark mode, the ebook reader shows dark grey text on a black background. Basically unreadable.",
        status="closed",
        days_ago=80,
        resolution="Reported to the mobile team; a contrast fix shipped in the next app update.",
    ),
    dict(
        category="technical",
        subject="Wishlist button does nothing",
        body="Tapping the heart/wishlist icon on a book page doesn't do anything, on both iOS and the website.",
        status="open",
        days_ago=2,
    ),
    dict(
        category="feature_request",
        subject="Please add an audiobook format",
        body="Would love to see audiobook versions of your fiction catalog, not just ebook and physical.",
        status="open",
        days_ago=10,
    ),
    dict(
        category="feature_request",
        subject="Bulk order discount for book clubs",
        body="I run a book club and often order 10+ copies of the same title. Any chance of a bulk-order discount option?",
        status="open",
        days_ago=25,
    ),
    dict(
        category="feature_request",
        subject="Add a 'notify me when back in stock' option",
        body="Several technology titles are out of stock for weeks at a time -- could you add a restock notification?",
        status="open",
        days_ago=15,
    ),
    dict(
        category="feature_request",
        subject="Gift wrapping option at checkout",
        body="It'd be great to have a gift-wrap checkbox at checkout for books I'm sending as presents.",
        status="resolved",
        days_ago=60,
        resolution="Passed along to product team as a formal feature request; added to the Q4 roadmap review list and closed the loop with the customer.",
    ),
    dict(
        category="feature_request",
        subject="Adjustable font size in the ebook reader",
        body="The ebook reader only has two font sizes. Could you add a slider for more control?",
        status="open",
        days_ago=7,
    ),
    dict(
        category="feature_request",
        subject="Reading progress sync across devices",
        body="My reading progress on the app doesn't sync with the web reader. Could that be added?",
        status="closed",
        days_ago=95,
        resolution="Logged as a longstanding feature request; still not implemented, closed as a duplicate of an existing engineering backlog item.",
    ),
    dict(
        category="feature_request",
        subject="Self-help workbook progress tracker",
        body="For workbook-style self-help titles, a built-in progress tracker (checkboxes per exercise) would be amazing.",
        status="open",
        days_ago=4,
    ),
]


def insert_new_tickets(conn, cat_ids, next_id):
    cur = conn.cursor()
    rng = random.Random(99)
    tid = next_id
    for spec in NEW_TICKETS:
        customer_id = rng.choice(_eligible_customers)
        created = shift(DATASET_AS_OF, days=-spec["days_ago"])
        updated = created
        if spec["status"] in ("resolved", "closed"):
            updated = shift(created, days=min(spec["days_ago"], 3))
        cur.execute(
            "INSERT INTO tickets (id, customer_id, category_id, subject, body, status, "
            "created_at, updated_at, book_category, format) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            (
                tid,
                customer_id,
                cat_ids[spec["category"]],
                spec["subject"],
                spec["body"],
                spec["status"],
                created,
                updated,
            ),
        )
        if spec["status"] in ("resolved", "closed"):
            cur.execute(
                "INSERT INTO ticket_history (customer_id, ticket_id, summary, resolved_at) VALUES (?, ?, ?, ?)",
                (customer_id, tid, spec["resolution"], updated),
            )
            cur.execute(
                "INSERT INTO replies (ticket_id, body, status, created_at, approved_at) VALUES (?, ?, 'approved', ?, ?)",
                (tid, spec["resolution"], shift(updated, hours=-1), updated),
            )
        else:
            cur.execute(
                "INSERT INTO replies (ticket_id, body, status, created_at, approved_at) VALUES (?, ?, 'proposed', ?, NULL)",
                (
                    tid,
                    "Thanks for the suggestion / heads up -- looking into this and will follow up soon.",
                    shift(created, hours=2),
                ),
            )
        tid += 1
    conn.commit()
    return tid - 1


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(db_path=DB_PATH):
    import os

    if os.path.exists(db_path):
        os.remove(db_path)
    conn = init_db(db_path)
    insert_customers(conn)
    cat_ids = category_ids(conn)
    final_status = insert_tickets(conn, cat_ids)
    insert_history_and_escalations_and_replies(conn, final_status)
    last_id = insert_new_tickets(conn, cat_ids, next_id=len(src.TICKETS) + 1)
    conn.commit()
    return conn, final_status, last_id


# ---------------------------------------------------------------------------
# Sanity checks -- every fact golden_eval_set.py depends on must still hold
# ---------------------------------------------------------------------------


def verify(conn, final_status):
    def status_counts(cid):
        rows = conn.execute(
            "SELECT status, COUNT(*) c FROM tickets WHERE customer_id=? GROUP BY status",
            (cid,),
        ).fetchall()
        return {r["status"]: r["c"] for r in rows}

    assert status_counts(3) == {"open": 2, "resolved": 2, "escalated": 1}, (
        status_counts(3)
    )
    assert status_counts(5) == {"open": 3, "resolved": 1, "escalated": 1}, (
        status_counts(5)
    )
    assert status_counts(7) == {"open": 2, "resolved": 3, "escalated": 1}, (
        status_counts(7)
    )
    assert status_counts(10) == {}, status_counts(10)
    assert status_counts(15) == {"resolved": 3, "escalated": 1}, status_counts(15)

    t5 = conn.execute("SELECT * FROM tickets WHERE id=5").fetchone()
    assert t5["customer_id"] == 2 and t5["status"] == "open"

    t45 = conn.execute("SELECT * FROM tickets WHERE id=45").fetchone()
    assert t45["customer_id"] == 12 and t45["status"] == "open"

    for cid in (4, 10):
        n = conn.execute(
            "SELECT COUNT(*) c FROM tickets WHERE customer_id=?", (cid,)
        ).fetchone()["c"]
        assert n == 0, (cid, n)

    open7 = conn.execute(
        "SELECT COUNT(*) c FROM tickets WHERE customer_id=7 AND status='open'"
    ).fetchone()["c"]
    open5 = conn.execute(
        "SELECT COUNT(*) c FROM tickets WHERE customer_id=5 AND status='open'"
    ).fetchone()["c"]
    assert open5 > open7, (open5, open7)

    # customer 7's plan must still carry the refund-window bonus ticket #30 relies on
    plan7 = conn.execute("SELECT plan FROM customers WHERE id=7").fetchone()["plan"]
    assert PLAN_REFUND_BONUS_DAYS[plan7] == 7, plan7

    injected_ids = {
        r["ticket_id"]
        for r in conn.execute("SELECT id ticket_id FROM tickets WHERE id IN (25,26,27)")
    }
    assert injected_ids == {25, 26, 27}

    n_replies = conn.execute("SELECT COUNT(*) c FROM replies").fetchone()["c"]
    n_history = conn.execute("SELECT COUNT(*) c FROM ticket_history").fetchone()["c"]
    n_escalations = conn.execute("SELECT COUNT(*) c FROM escalations").fetchone()["c"]
    n_tickets = conn.execute("SELECT COUNT(*) c FROM tickets").fetchone()["c"]
    print(
        f"tickets={n_tickets} replies={n_replies} ticket_history={n_history} escalations={n_escalations}"
    )

    print("All protected facts verified OK.")


def main():
    conn, final_status, last_id = build(DB_PATH)
    verify(conn, final_status)
    print(f"Wrote {DB_PATH} (ticket ids 1..{last_id}).")
    conn.close()


if __name__ == "__main__":
    main()
