"""Data-quality rules expressed as SQL predicates.

The same rule dicts drive two runtimes:
  * locally, `split_valid` keeps passing rows and routes failures to a quarantine table
    together with the names of the rules they broke;
  * on Databricks, dlt_pipeline.py passes them to `dlt.expect_all_or_drop`.
"""
from __future__ import annotations

from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..config import FX_TO_USD

CUSTOMER_RULES = {
    "customer_id_format": "customer_id RLIKE '^C[0-9]{4}$'",
    "region_valid": "region IN ('NA','EMEA','APAC','LATAM')",
    "segment_valid": "segment IN ('enterprise','midmarket','smb')",
    "updated_at_valid": "updated_at IS NOT NULL",
    "email_has_at": "email LIKE '%@%'",
}

_CCY = ",".join(f"'{c}'" for c in FX_TO_USD)
INVOICE_RULES = {
    "invoice_id_present": "invoice_id IS NOT NULL AND invoice_id <> ''",
    "amount_not_null": "amount IS NOT NULL",
    "amount_positive": "amount > 0",
    "currency_valid": f"currency IN ({_CCY})",
    "status_valid": "status IN ('paid','open','void')",
    "issued_at_valid": "issued_at IS NOT NULL",
}

TICKET_RULES = {
    "ticket_id_present": "ticket_id IS NOT NULL AND ticket_id <> ''",
    "priority_valid": "priority IN ('P1','P2','P3')",
    "status_valid": "status IN ('open','closed','pending')",
    "opened_at_valid": "opened_at IS NOT NULL",
}


def split_valid(df: DataFrame, rules: dict[str, str]) -> tuple[DataFrame, DataFrame]:
    """Return (valid, quarantined). A NULL predicate result counts as a failure (fail closed)."""
    passed = {name: F.coalesce(F.expr(expr), F.lit(False)) for name, expr in rules.items()}
    all_ok = reduce(lambda a, b: a & b, passed.values())
    failed = F.array_compact(F.array(*[F.when(~ok, F.lit(n)) for n, ok in passed.items()]))
    valid = df.where(all_ok)
    quarantined = df.where(~all_ok).withColumn("_failed_rules", F.array_join(failed, ","))
    return valid, quarantined
