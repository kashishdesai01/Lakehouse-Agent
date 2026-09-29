import pytest

from lakehouse_agent.agent.tools import SqlGuardError, validate_sql


@pytest.mark.parametrize("q", [
    "SELECT region FROM monthly_revenue",
    "SELECT segment, COUNT(*) FROM customer_360 GROUP BY segment",
    "WITH t AS (SELECT * FROM ticket_summary) SELECT SUM(ticket_count) FROM t",
    "SELECT a.region FROM customer_360 a JOIN monthly_revenue b ON a.region = b.region",
])
def test_allows_read_only_gold_queries(q):
    assert validate_sql(q)


@pytest.mark.parametrize("q", [
    "DROP TABLE customer_360",
    "DELETE FROM customer_360",
    "INSERT INTO customer_360 VALUES (1)",
    "UPDATE customer_360 SET region = 'NA'",
    "SELECT * FROM customer_360; DROP TABLE customer_360",
    "SELECT * FROM silver_customers",
    "SELECT * FROM main.default.customer_360",
    "CREATE TABLE x AS SELECT * FROM customer_360",
    "SELEC oops",
])
def test_rejects_writes_multi_statement_and_other_tables(q):
    with pytest.raises(SqlGuardError):
        validate_sql(q)
