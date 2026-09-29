#!/usr/bin/env bash
# Build the wheel, validate and deploy the Databricks bundle, then run the job.
# Requires DATABRICKS_HOST and an authenticated Databricks CLI.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DATABRICKS_HOST:?set DATABRICKS_HOST to your workspace URL}"

python -m pip install --quiet build
python -m build --wheel

databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run lakehouse_agent_job -t dev

echo "Done. Open the run in the workspace UI and copy the job run URL and MLflow experiment URL into the README."
