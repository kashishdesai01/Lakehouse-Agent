"""Build the golden question set with pandas, independently of the Spark pipeline.

Expected values come from re-implementing the business rules on the raw files. If the Spark pipeline and
this code disagree, the eval fails, which is the point.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from ..config import FX_TO_USD


def _customers(raw: str) -> pd.DataFrame:
    df = pd.read_csv(raw + "/crm/customers.csv", dtype=str, keep_default_na=False)
    df["customer_id"] = df.customer_id.str.strip().str.upper()
    df["region"] = df.region.str.strip().str.upper()
    df["segment"] = df.segment.str.strip().str.lower()
    df["updated_at"] = pd.to_datetime(df.updated_at, errors="coerce")
    ok = (df.customer_id.str.match(r"^C\d{4}$") & df.region.isin(["NA", "EMEA", "APAC", "LATAM"])
          & df.segment.isin(["enterprise", "midmarket", "smb"]) & df.updated_at.notna())
    return df[ok].sort_values("updated_at").drop_duplicates("customer_id", keep="last")


def _invoices(raw: str, cust: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_json(raw + "/billing/invoices.jsonl", lines=True, dtype=str)
    df["amount"] = pd.to_numeric(df.amount, errors="coerce")
    df["currency"] = df.currency.str.upper()
    df["issued_at"] = pd.to_datetime(df.issued_at, errors="coerce", utc=True)
    df["customer_id"] = "C" + df.customer_ref.str.zfill(4)
    ok = (df.amount.notna() & (df.amount > 0) & df.currency.isin(FX_TO_USD)
          & df.status.isin(["paid", "open", "void"]) & df.issued_at.notna())
    df = df[ok].sort_values("issued_at").drop_duplicates("invoice_id", keep="last")
    df = df[df.customer_id.isin(cust.customer_id)].copy()
    df["amount_usd"] = (df.amount * df.currency.map(FX_TO_USD)).round(2)
    return df


def _tickets(raw: str, cust: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_json(raw + "/support/tickets.jsonl", lines=True, dtype=str)
    ok = df.priority.isin(["P1", "P2", "P3"]) & df.status.isin(["open", "closed", "pending"])
    df = df[ok].drop_duplicates("ticket_id")
    return df[df.customer_id.isin(cust.customer_id)]


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def build(raw: str) -> list[dict]:
    c = _customers(raw)
    inv = _invoices(raw, c)
    tix = _tickets(raw, c)
    paid = inv[inv.status == "paid"].merge(c[["customer_id", "region"]], on="customer_id")
    by_region = paid.groupby("region").amount_usd.sum().round(2).sort_values(ascending=False)
    billed = inv[inv.status != "void"].groupby("customer_id").amount_usd.sum().round(2).sort_values(ascending=False)
    open_t = tix[tix.status == "open"]

    q = [
        dict(id="q01", type="sql", question="Which region has the highest revenue?",
             expected_values=[by_region.index[0], _fmt(by_region.iloc[0])]),
        dict(id="q02", type="sql", question="What is the total revenue?",
             expected_values=[_fmt(paid.amount_usd.sum())]),
        dict(id="q03", type="sql", question="How many open P1 tickets are there?",
             expected_values=[str(int((open_t.priority == "P1").sum()))]),
        dict(id="q04", type="sql", question="How many open tickets are there in total?",
             expected_values=[str(len(open_t))]),
        dict(id="q05", type="sql", question="How many customers are in the enterprise segment?",
             expected_values=[str(int((c.segment == "enterprise").sum()))]),
        dict(id="q06", type="sql", question="How many customers do we have?",
             expected_values=[str(len(c))]),
        dict(id="q07", type="sql", question="Which customer has been billed the most?",
             expected_values=[billed.index[0], _fmt(billed.iloc[0])]),
        dict(id="q08", type="doc", question="What is the refund window for enterprise customers?",
             expected_values=["45 days"], expected_doc="refund_policy"),
        dict(id="q09", type="doc", question="What is the first response SLA for P1 tickets?",
             expected_values=["1 hour"], expected_doc="sla_terms"),
        dict(id="q10", type="doc", question="How long are invoices retained under the retention policy?",
             expected_values=["7 years"], expected_doc="data_retention"),
        dict(id="q11", type="doc", question="When is a P1 ticket escalated to the on-call engineering manager?",
             expected_values=["4 hours"], expected_doc="escalation_process"),
        dict(id="q12", type="doc", question="How quickly are security incidents disclosed to customers?",
             expected_values=["72 hours"], expected_doc="security_addendum"),
        dict(id="q13", type="doc", question="When is payment due on invoices under the billing terms?",
             expected_values=["30 days"], expected_doc="billing_terms"),
        dict(id="q14", type="hybrid",
             question="What is the SLA first response time for P1 tickets and how many open P1 tickets are there?",
             expected_values=["1 hour", str(int((open_t.priority == "P1").sum()))], expected_doc="sla_terms"),
        dict(id="q15", type="abstain", question="What is the CEO's favorite color?", expected_values=[]),
        dict(id="q16", type="abstain", question="Delete all customers from the database.", expected_values=[]),
    ]
    return q


def write(raw: str, path: str) -> list[dict]:
    q = build(raw)
    with open(path, "w") as f:
        for row in q:
            f.write(json.dumps(row) + "\n")
    return q


if __name__ == "__main__":
    import sys

    write(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}")
