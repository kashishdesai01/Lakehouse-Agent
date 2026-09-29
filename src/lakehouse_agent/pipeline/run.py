"""Local/Databricks-job entrypoint: raw extracts -> bronze -> silver (+quarantine) -> gold."""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone

from pyspark.sql import Row, SparkSession

from ..config import Settings, add_settings_args, settings_from_args
from ..spark_session import get_spark
from ..storage import read_table, write_table
from . import bronze, gold, silver


def run(spark: SparkSession, raw: dict, s: Settings, batch_id: str | None = None) -> list[dict]:
    batch_id = batch_id or uuid.uuid4().hex[:12]
    log: list[dict] = []

    def record(stage: str, table: str, rows_in: int, rows_out: int, quarantined: int = 0) -> None:
        log.append({"batch_id": batch_id, "stage": stage, "table": table, "rows_in": rows_in,
                    "rows_out": rows_out, "rows_quarantined": quarantined,
                    "ts": datetime.now(timezone.utc).isoformat()})

    # bronze
    b_c = bronze.ingest_customers(spark, raw["crm"], batch_id)
    b_i = bronze.ingest_invoices(spark, raw["billing"], batch_id)
    b_t = bronze.ingest_tickets(spark, raw["support"], batch_id)
    for name, df in (("customers", b_c), ("invoices", b_i), ("tickets", b_t)):
        write_table(df, "bronze", name, s)
        n = read_table(spark, "bronze", name, s).count()
        record("bronze", name, n, n)

    b_c, b_i, b_t = (read_table(spark, "bronze", n, s) for n in ("customers", "invoices", "tickets"))

    # silver
    c_ok, c_bad = silver.clean_customers(b_c)
    write_table(c_ok, "silver", "customers", s)
    write_table(c_bad, "quarantine", "customers", s)
    c_ok = read_table(spark, "silver", "customers", s)

    i_ok, i_bad = silver.clean_invoices(b_i, c_ok)
    t_ok, t_bad = silver.clean_tickets(b_t, c_ok)
    write_table(i_ok, "silver", "invoices", s)
    write_table(i_bad, "quarantine", "invoices", s)
    write_table(t_ok, "silver", "tickets", s)
    write_table(t_bad, "quarantine", "tickets", s)
    for name, n_in in (("customers", b_c.count()), ("invoices", b_i.count()), ("tickets", b_t.count())):
        out = read_table(spark, "silver", name, s).count()
        bad = read_table(spark, "quarantine", name, s).count()
        record("silver", name, n_in, out, bad)

    # gold
    c, i, t = (read_table(spark, "silver", n, s) for n in ("customers", "invoices", "tickets"))
    for name, df in (("customer_360", gold.customer_360(c, i, t)),
                     ("monthly_revenue", gold.monthly_revenue(c, i)),
                     ("ticket_summary", gold.ticket_summary(c, t))):
        write_table(df, "gold", name, s)
        n = read_table(spark, "gold", name, s).count()
        record("gold", name, n, n)

    write_table(spark.createDataFrame([Row(**r) for r in log]), "ops", "pipeline_run_log", s, mode="append")
    return log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True, help="directory produced by data.synth.generate")
    add_settings_args(p)
    a = p.parse_args()
    s = settings_from_args(a)
    spark = get_spark(s)
    raw = {"crm": f"{a.raw_dir}/crm/customers.csv", "billing": f"{a.raw_dir}/billing/invoices.jsonl",
           "support": f"{a.raw_dir}/support/tickets.jsonl"}
    for r in run(spark, raw, s):
        print(r)


if __name__ == "__main__":
    main()
