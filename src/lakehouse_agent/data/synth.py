"""Deterministic synthetic source-system extracts (CRM, billing, support) plus policy documents.

The extracts are deliberately dirty (duplicates, bad amounts, inconsistent ids and casing) so the
silver layer has something real to reject. Seeded, so the golden set is reproducible.
"""
from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timedelta

REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
SEGMENTS = ["enterprise", "midmarket", "smb"]
CURRENCIES = ["USD", "EUR", "GBP"]
STATUSES = ["paid", "open", "void"]
PRIORITIES = ["P1", "P2", "P3"]
TICKET_STATUS = ["open", "closed", "pending"]

DOCS = {
    "refund_policy": (
        "Refund Policy\n\nStandard and midmarket customers may request a refund within 30 days of "
        "the invoice date. Enterprise customers may request a refund within 45 days of the invoice "
        "date. Refunds are issued to the original payment method within 10 business days. Void "
        "invoices are never refunded because no payment was collected."
    ),
    "sla_terms": (
        "Support SLA Terms\n\nPriority P1 tickets receive a first response within 1 hour, 24x7. "
        "Priority P2 tickets receive a first response within 4 hours during business hours. "
        "Priority P3 tickets receive a first response within 2 business days. Enterprise customers "
        "are entitled to a named technical account manager."
    ),
    "data_retention": (
        "Data Retention Policy\n\nCustomer invoices are retained for 7 years to meet audit "
        "requirements. Support ticket content is retained for 3 years and then anonymised. Raw "
        "ingestion files in the bronze layer are retained for 90 days."
    ),
    "escalation_process": (
        "Escalation Process\n\nA ticket is escalated to the on-call engineering manager when a P1 "
        "ticket remains unresolved for 4 hours. Customers with an enterprise segment are escalated "
        "automatically at 2 hours. All escalations are recorded in the ticket history."
    ),
    "billing_terms": (
        "Billing Terms\n\nInvoices are issued monthly in the customer's contracted currency. "
        "Payment is due within 30 days of issue. Invoices unpaid after 60 days are marked overdue "
        "and may trigger service suspension for smb and midmarket accounts."
    ),
    "security_addendum": (
        "Security Addendum\n\nAll customer data is encrypted at rest with AES-256 and in transit "
        "with TLS 1.2 or higher. Access to production data requires multi-factor authentication "
        "and is logged. Security incidents are disclosed to affected customers within 72 hours."
    ),
}


def _ts(rng: random.Random, start: datetime, days: int) -> datetime:
    return start + timedelta(days=rng.randrange(days), hours=rng.randrange(24))


def generate(out_dir: str, n_customers: int = 60, seed: int = 7) -> dict:
    """Write raw extracts under out_dir and return the paths."""
    rng = random.Random(seed)
    start = datetime(2026, 1, 1)
    os.makedirs(os.path.join(out_dir, "crm"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "billing"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "support"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "docs"), exist_ok=True)

    # CRM: customer ids are "C0001". Some customers appear twice (an older and a newer revision),
    # with messy casing/whitespace.
    crm_rows = []
    for i in range(1, n_customers + 1):
        cid = f"C{i:04d}"
        region = rng.choice(REGIONS)
        seg = rng.choice(SEGMENTS)
        name = f"Customer {i}"
        email = f"contact{i}@example.com"
        updated = _ts(rng, start, 200)
        crm_rows.append([cid, name, email, region, seg, updated.strftime("%Y-%m-%d %H:%M:%S")])
        if i % 9 == 0:  # older duplicate with stale region
            older = updated - timedelta(days=30)
            crm_rows.append([cid, name.upper(), f" {email.upper()} ", rng.choice(REGIONS), seg,
                             older.strftime("%Y-%m-%d %H:%M:%S")])
    crm_rows.append(["", "No Id Customer", "x@example.com", "NA", "smb", "2026-02-01 00:00:00"])  # bad
    crm_rows.append(["C9999", "Bad Region", "b@example.com", "MARS", "smb", "2026-02-01 00:00:00"])  # bad
    with open(os.path.join(out_dir, "crm", "customers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "full_name", "email", "region", "segment", "updated_at"])
        w.writerows(crm_rows)

    # Billing: references customers as "0001" (no prefix) - silver must reconcile the id format.
    inv = []
    n = 0
    for i in range(1, n_customers + 1):
        for _ in range(rng.randrange(1, 6)):
            n += 1
            inv.append({
                "invoice_id": f"INV-{n:05d}",
                "customer_ref": f"{i:04d}",
                "amount": str(round(rng.uniform(50, 5000), 2)),
                "currency": rng.choice(CURRENCIES),
                "status": rng.choices(STATUSES, weights=[6, 3, 1])[0],
                "issued_at": _ts(rng, start, 240).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
    inv.append({**inv[0]})  # exact duplicate delivery
    inv.append({"invoice_id": "INV-BAD1", "customer_ref": "0001", "amount": None, "currency": "USD",
                "status": "paid", "issued_at": "2026-03-01T00:00:00Z"})
    inv.append({"invoice_id": "INV-BAD2", "customer_ref": "0002", "amount": "-40.00", "currency": "USD",
                "status": "paid", "issued_at": "2026-03-01T00:00:00Z"})
    inv.append({"invoice_id": "INV-BAD3", "customer_ref": "0003", "amount": "10.00", "currency": "XXX",
                "status": "paid", "issued_at": "2026-03-01T00:00:00Z"})
    with open(os.path.join(out_dir, "billing", "invoices.jsonl"), "w") as f:
        for r in inv:
            f.write(json.dumps(r) + "\n")

    # Support tickets reference customers as "C0001".
    tix = []
    t = 0
    for i in range(1, n_customers + 1):
        for _ in range(rng.randrange(0, 5)):
            t += 1
            tix.append({
                "ticket_id": f"T-{t:05d}",
                "customer_id": f"C{i:04d}",
                "subject": rng.choice(["Login failure", "Invoice question", "Data export", "Outage"]),
                "priority": rng.choices(PRIORITIES, weights=[2, 4, 4])[0],
                "status": rng.choice(TICKET_STATUS),
                "opened_at": _ts(rng, start, 240).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
    tix.append({"ticket_id": "T-BAD1", "customer_id": "C0001", "subject": "x", "priority": "P9",
                "status": "open", "opened_at": "2026-03-01T00:00:00Z"})
    with open(os.path.join(out_dir, "support", "tickets.jsonl"), "w") as f:
        for r in tix:
            f.write(json.dumps(r) + "\n")

    for doc_id, text in DOCS.items():
        with open(os.path.join(out_dir, "docs", f"{doc_id}.md"), "w") as f:
            f.write(text)

    return {
        "crm": os.path.join(out_dir, "crm", "customers.csv"),
        "billing": os.path.join(out_dir, "billing", "invoices.jsonl"),
        "support": os.path.join(out_dir, "support", "tickets.jsonl"),
        "docs": os.path.join(out_dir, "docs"),
    }


def main() -> None:
    """Write raw extracts, documents and the golden question set (used by the Databricks seed task)."""
    import argparse

    from ..evals import build_golden

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    generate(a.out_dir)
    build_golden.write(a.out_dir, os.path.join(a.out_dir, "golden.jsonl"))
    print(f"wrote raw extracts, docs and golden set under {a.out_dir}")


if __name__ == "__main__":
    main()
