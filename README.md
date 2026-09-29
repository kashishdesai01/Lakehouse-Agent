# Lakehouse Agent

A medallion data pipeline (bronze, silver, gold) with a tool-using agent on top and an LLM evaluation harness.
Built for Databricks on Azure (Delta Lake, Unity Catalog, Delta Live Tables, MLflow), and runnable locally on
plain Spark so the logic can be tested without a cluster.

**What it does.** Three source systems (CRM CSV, billing JSON, support-ticket JSON) are landed, cleaned,
deduplicated and modelled into gold tables. An agent answers questions by choosing between a guarded SQL tool
over the gold tables and a retrieval tool over policy documents, or by combining both. An evaluation harness
scores it on retrieval accuracy, answer correctness and hallucination rate.

```
raw extracts ──► bronze (all strings + lineage cols) ──► silver (typed, deduped) ──► gold (customer_360,
   CRM/billing/tickets                  │                     │                        monthly_revenue,
                                        │                     └─► quarantine            ticket_summary)
policy docs ──► chunks ──► retriever ───┴───────────────────────────────────────────────────┐
                                                                                            ▼
                                            Agent loop ── run_sql (SELECT-only, allow-listed tables)
                                                       └─ search_documents (cited chunks)
                                                            │
                                                     MLflow traces + eval metrics
```

## Tested and verified

The test suite runs the complete local workflow on Spark and checks the Azure and Databricks deployment definitions:
**32 tests pass** (`pytest`).

| Area | Verified by |
|---|---|
| Bronze keeps every row with `_source_file`, `_ingested_at`, `_batch_id` | tests/test_pipeline.py |
| Silver rules quarantine bad rows and name the broken rule; nothing vanishes (row counts reconcile) | test_pipeline.py |
| CRM dedup keeps latest revision; billing ids reconciled to CRM id format; FX to USD | test_pipeline.py |
| Gold numbers match an **independent pandas implementation** of the same rules | test_pipeline.py + evals/build_golden.py |
| Re-running the pipeline does not duplicate gold | test_pipeline.py |
| SQL guard rejects writes, DDL, multi-statement, non-gold tables, qualified names | tests/test_sql_guard.py |
| Agent loop: hybrid questions use both tools, step limit stops loops, tool errors are fed back | tests/test_agent_eval.py |
| Harness catches bad agents (see below) and logs runs to MLflow | tests/test_agent_eval.py |
| Databricks bundle contains the seed -> medallion -> evaluate job graph and DLT pipeline | tests/test_deployment_config.py |
| Terraform parses and defines the Azure resource group, ADLS Gen2 storage and Premium Databricks workspace | tests/test_deployment_config.py |
| Deployment script passes Bash syntax validation and invokes bundle validation, deployment and execution | tests/test_deployment_config.py |

Local execution uses Parquet through the same storage abstraction, TF-IDF retrieval and a deterministic model stand-in.
The Databricks target switches storage to Delta and Unity Catalog, includes a Delta Live Tables pipeline and supports
Mosaic AI Vector Search plus OpenAI-compatible model endpoints.

## Evaluation: what the numbers mean

`evals/build_golden.py` produces 16 questions (7 SQL, 6 document, 1 hybrid, 2 that must be refused). Metrics:

- `answer_accuracy`: expected numbers/facts appear in the answer.
- `retrieval_hit_rate` / `retrieval_top1_rate`: the expected document was retrieved (anywhere / first).
- `hallucination_rate`: share of answers stating a number or citing a document that no tool result supports.
- `abstain_accuracy`: unanswerable or out-of-scope requests are declined.

Measured on the seeded data:

| Agent | accuracy | abstain | hallucination |
|---|---|---|---|
| `RuleBasedLLM` (offline router) | 1.00 | 1.00 | 0.00 |
| `HallucinatingLLM` (negative control) | 0.125 | 1.00 | 0.875 |
| `OverconfidentLLM` (negative control) | 0.875 | 0.00 | 0.125 |

The rule-based 1.00 only shows the plumbing works; it is a keyword router and says nothing about model quality.
The negative controls are the meaningful result: they show the harness fails agents that fabricate or over-answer.
The hallucination metric is a groundedness check on numbers and citations, not a judge of reasoning. The golden set
is small (16) and synthetic. Run `lh-eval --llm openai --model <endpoint>` against a real model to get real numbers.

## Azure Databricks deployment

The repository includes Terraform for the Azure resource group, ADLS Gen2 account and Premium Databricks workspace,
plus a Databricks Asset Bundle for the three-stage workflow. The deployment script builds the wheel, validates the
bundle, deploys it and starts the job. See [DEPLOY.md](DEPLOY.md) for the commands and required account setup.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q

lh-seed --out-dir /tmp/lh/raw                           # extracts, docs, golden.jsonl
lh-pipeline --raw-dir /tmp/lh/raw --root /tmp/lh/lake   # bronze -> silver -> gold
lh-eval --raw-dir /tmp/lh/raw --golden /tmp/lh/raw/golden.jsonl --root /tmp/lh/lake
```

On Databricks: `terraform -chdir=infra apply`, then `databricks bundle deploy -t dev` and
`databricks bundle run lakehouse_agent_job -t dev` (Unity Catalog mode is `--storage-mode uc`).

## Design decisions worth explaining

- **Safety is structural.** The SQL tool parses with sqlglot and accepts exactly one SELECT over an allow-list of
  gold tables; the destructive-query test shows the table is untouched. It does not rely on prompt wording.
- **Nothing is silently dropped.** Rule failures go to `quarantine_*` with the failing rule names; every stage
  writes counts to `ops_pipeline_run_log`.
- **Same rules, two runtimes.** Expectations are SQL predicate dicts used by the local runner and by
  `dlt.expect_all_or_drop`, so they cannot drift apart.
- **Independent oracle.** Expected answers come from a separate pandas implementation, so a bug in the Spark
  transforms shows up as an eval failure instead of being baked into the expected values.
- **Known limits.** Fixed FX rates, TF-IDF retrieval locally, single-node Spark in tests, synthetic data.
