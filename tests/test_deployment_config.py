from pathlib import Path
import subprocess

import hcl2
import yaml


ROOT = Path(__file__).parents[1]


def test_databricks_bundle_defines_the_full_job_graph():
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())

    assert bundle["bundle"]["name"] == "lakehouse-agent"
    job = bundle["resources"]["jobs"]["lakehouse_agent_job"]
    tasks = {task["task_key"]: task for task in job["tasks"]}

    assert set(tasks) == {"seed", "medallion", "evaluate"}
    assert tasks["medallion"]["depends_on"] == [{"task_key": "seed"}]
    assert tasks["evaluate"]["depends_on"] == [{"task_key": "medallion"}]
    assert "lakehouse_dlt" in bundle["resources"]["pipelines"]


def test_terraform_defines_azure_lakehouse_resources():
    with (ROOT / "infra" / "main.tf").open() as config_file:
        config = hcl2.load(config_file)

    resources = {
        resource_type.strip('"'): {
            name.strip('"'): attributes for name, attributes in instances.items()
        }
        for block in config["resource"]
        for resource_type, instances in block.items()
    }

    assert "azurerm_resource_group" in resources
    assert "azurerm_storage_account" in resources
    assert "azurerm_storage_container" in resources
    assert "azurerm_databricks_workspace" in resources
    storage = resources["azurerm_storage_account"]["lake"]
    workspace = resources["azurerm_databricks_workspace"]["dbx"]
    assert storage["is_hns_enabled"] is True
    assert workspace["sku"].strip('"') == "premium"


def test_deployment_script_is_valid_bash_and_runs_bundle_validation():
    script = ROOT / "scripts" / "deploy.sh"

    subprocess.run(["bash", "-n", script], check=True)
    contents = script.read_text()
    assert "databricks bundle validate -t dev" in contents
    assert "databricks bundle deploy -t dev" in contents
    assert "databricks bundle run lakehouse_agent_job -t dev" in contents
