from lakehouse_agent.storage import read_table


def _table(env, layer, name):
    return read_table(env["spark"], layer, name, env["s"])


def test_no_row_is_silently_lost(env):
    """Every bronze row is either in silver, in quarantine, or an explained duplicate."""
    by = {(r["stage"], r["table"]): r for r in env["log"]}
    assert by[("silver", "invoices")]["rows_in"] == 174
    assert by[("silver", "invoices")]["rows_out"] == 170
    assert by[("silver", "invoices")]["rows_quarantined"] == 3   # INV-BAD1..3
    # 174 - 170 - 3 = 1 exact duplicate delivery dropped by dedup
    assert by[("silver", "tickets")]["rows_quarantined"] == 1
    assert by[("silver", "customers")]["rows_quarantined"] == 2


def test_quarantine_names_the_broken_rules(env):
    rows = {r["invoice_id"]: r["_failed_rules"] for r in _table(env, "quarantine", "invoices").collect()}
    assert "amount_not_null" in rows["INV-BAD1"]
    assert rows["INV-BAD2"] == "amount_positive"
    assert rows["INV-BAD3"] == "currency_valid"
    bad_tix = _table(env, "quarantine", "tickets").collect()
    assert bad_tix[0]["_failed_rules"] == "priority_valid"


def test_customer_dedup_keeps_latest_revision(env):
    c = _table(env, "silver", "customers")
    assert c.count() == c.select("customer_id").distinct().count() == 60
    # customers with an older stale duplicate: id C0009 has both; silver must hold one clean row
    row = c.where("customer_id = 'C0009'").collect()[0]
    assert row["email"] == row["email"].strip().lower()
    assert row["full_name"] == "Customer 9"


def test_billing_ids_reconcile_to_crm_format(env):
    i = _table(env, "silver", "invoices")
    assert i.where("NOT customer_id RLIKE '^C[0-9]{4}$'").count() == 0


def test_bronze_keeps_everything_with_lineage(env):
    b = _table(env, "bronze", "invoices")
    assert b.count() == 174
    assert {"_source_file", "_ingested_at", "_batch_id"} <= set(b.columns)
    assert b.where("_batch_id = 'test'").count() == 174


def test_gold_matches_independent_pandas_numbers(env):
    """The pandas golden builder re-implements the rules; Spark gold must agree with it."""
    g = {q["id"]: q for q in env["golden"]}
    rev = _table(env, "gold", "monthly_revenue")
    total = rev.agg({"revenue_usd": "sum"}).collect()[0][0]
    assert abs(total - float(g["q02"]["expected_values"][0])) < 0.05
    top = rev.groupBy("region").sum("revenue_usd").orderBy("sum(revenue_usd)", ascending=False).first()
    assert top["region"] == g["q01"]["expected_values"][0]
    c360 = _table(env, "gold", "customer_360")
    assert c360.count() == int(g["q06"]["expected_values"][0])
    assert c360.where("segment = 'enterprise'").count() == int(g["q05"]["expected_values"][0])
    open_p1 = c360.agg({"open_p1_count": "sum"}).collect()[0][0]
    assert open_p1 == int(g["q03"]["expected_values"][0])


def test_pipeline_is_rerunnable_without_duplicating_gold(env):
    from lakehouse_agent.pipeline.run import run

    p = env["paths"]
    before = _table(env, "gold", "customer_360").count()
    run(env["spark"], {k: p[k] for k in ("crm", "billing", "support")}, env["s"], batch_id="again")
    assert _table(env, "gold", "customer_360").count() == before
