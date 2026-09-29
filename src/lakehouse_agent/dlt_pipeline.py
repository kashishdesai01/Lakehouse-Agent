"""Delta Live Tables definition of the same medallion flow for the Databricks runtime.

It reuses the typing/dedup logic and the rule dicts from the local pipeline, so a rule change lands in one
place. Set the pipeline configuration key `raw_dir` to the ADLS Gen2 path or UC volume holding the extracts.
"""
import dlt  # noqa: F401  (provided by the Databricks runtime)
from pyspark.sql import functions as F

from lakehouse_agent.pipeline import gold as gold_mod
from lakehouse_agent.pipeline.expectations import CUSTOMER_RULES, INVOICE_RULES, TICKET_RULES
from lakehouse_agent.pipeline.silver import _latest, add_usd, type_customers, type_invoices, type_tickets

RAW = spark.conf.get("raw_dir")  # noqa: F821  (spark is injected by DLT)
_STR = "customer_id string, full_name string, email string, region string, segment string, updated_at string"


def _lineage(df):
    return (df.withColumn("_source_file", F.col("_metadata.file_path"))
              .withColumn("_ingested_at", F.current_timestamp())
              .withColumn("_batch_id", F.lit(spark.conf.get("pipelines.id", "dlt"))))  # noqa: F821


@dlt.table(name="bronze_customers", comment="Raw CRM extract, all strings")
def bronze_customers():
    return _lineage(spark.readStream.format("cloudFiles").option("cloudFiles.format", "csv")  # noqa: F821
                    .option("header", True).schema(_STR).load(f"{RAW}/crm"))


@dlt.table(name="bronze_invoices", comment="Raw billing extract, all strings")
def bronze_invoices():
    schema = "invoice_id string, customer_ref string, amount string, currency string, status string, issued_at string"
    return _lineage(spark.readStream.format("cloudFiles").option("cloudFiles.format", "json")  # noqa: F821
                    .schema(schema).load(f"{RAW}/billing"))


@dlt.table(name="bronze_tickets", comment="Raw support extract, all strings")
def bronze_tickets():
    schema = "ticket_id string, customer_id string, subject string, priority string, status string, opened_at string"
    return _lineage(spark.readStream.format("cloudFiles").option("cloudFiles.format", "json")  # noqa: F821
                    .schema(schema).load(f"{RAW}/support"))


@dlt.view(name="typed_customers")
@dlt.expect_all_or_drop(CUSTOMER_RULES)
def typed_customers():
    return type_customers(dlt.read("bronze_customers"))


@dlt.table(name="silver_customers", comment="Latest valid revision per customer")
def silver_customers():
    return _latest(dlt.read("typed_customers"), "customer_id", "updated_at")


@dlt.view(name="typed_invoices")
@dlt.expect_all_or_drop(INVOICE_RULES)
def typed_invoices():
    return type_invoices(dlt.read("bronze_invoices"))


@dlt.table(name="silver_invoices", comment="Deduplicated valid invoices with USD amounts")
@dlt.expect_or_drop("customer_exists", "customer_id IS NOT NULL")
def silver_invoices():
    known = _latest(dlt.read("typed_invoices"), "invoice_id", "issued_at").join(
        dlt.read("silver_customers").select("customer_id"), "customer_id", "left_semi")
    return add_usd(known)


@dlt.view(name="typed_tickets")
@dlt.expect_all_or_drop(TICKET_RULES)
def typed_tickets():
    return type_tickets(dlt.read("bronze_tickets"))


@dlt.table(name="silver_tickets", comment="Deduplicated valid tickets")
def silver_tickets():
    return _latest(dlt.read("typed_tickets"), "ticket_id", "opened_at").join(
        dlt.read("silver_customers").select("customer_id"), "customer_id", "left_semi")


@dlt.table(name="gold_customer_360")
def gold_customer_360():
    return gold_mod.customer_360(dlt.read("silver_customers"), dlt.read("silver_invoices"), dlt.read("silver_tickets"))


@dlt.table(name="gold_monthly_revenue")
def gold_monthly_revenue():
    return gold_mod.monthly_revenue(dlt.read("silver_customers"), dlt.read("silver_invoices"))


@dlt.table(name="gold_ticket_summary")
def gold_ticket_summary():
    return gold_mod.ticket_summary(dlt.read("silver_customers"), dlt.read("silver_tickets"))
