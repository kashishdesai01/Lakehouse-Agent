# Deploying to Azure Databricks

This runbook provisions the Azure resources, creates the Unity Catalog objects and deploys the Databricks Asset
Bundle. Automated tests parse the Terraform and bundle definitions, verify the job dependency graph and validate
the deployment script's Bash syntax.

## Prerequisites
- Azure subscription (free credit works) and `az` CLI logged in: `az login`
- `terraform` >= 1.5 and the Databricks CLI (`databricks`) >= 0.230
- Python 3.10+ with `pip install build`

## 1. Provision Azure (about 5 minutes)
```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars   # edit prefix/location; prefix must be globally unique
terraform -chdir=infra init
terraform -chdir=infra apply
```
Outputs: `workspace_url` and `raw_path` (ADLS Gen2). Premium SKU is used so Unity Catalog lineage is available.

## 2. Unity Catalog objects
Open the workspace SQL editor and run `infra/uc_setup.sql` (catalog, schema, raw volume).
If your account has no metastore attached yet, attach one in the Databricks account console first.

## 3. Deploy and run the bundle
```bash
export DATABRICKS_HOST=https://<workspace_url from step 1>
databricks auth login --host "$DATABRICKS_HOST"
scripts/deploy.sh          # builds the wheel, validates, deploys, runs the job
```
The job runs three tasks: `seed` (writes synthetic extracts, docs and the golden set to the UC volume),
`medallion` (bronze/silver/gold Delta tables in Unity Catalog), `evaluate` (eval harness logged to MLflow).

To run the Delta Live Tables version instead: `databricks bundle run lakehouse_dlt -t dev`.

## 4. Real model for the eval (optional)
Create or pick a Model Serving endpoint, then rerun the evaluate task with `--llm openai --model <endpoint>` and
set `LLM_BASE_URL` / `LLM_API_KEY`. Without this the eval uses the offline stand-in and only validates plumbing.

## 5. Verify the run

Open the completed job in Databricks and check the three task outputs, the MLflow experiment at
`/Shared/lakehouse-agent-eval`, the tables under `main.lakehouse_agent` and the Unity Catalog lineage view.
