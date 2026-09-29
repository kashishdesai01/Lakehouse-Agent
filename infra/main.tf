# Azure footprint for the lakehouse: ADLS Gen2 for raw/bronze data and an Azure Databricks workspace.
# The configuration structure is covered by tests/test_deployment_config.py.
terraform {
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 3.100" }
  }
}

provider "azurerm" {
  features {}
}

variable "location" { default = "eastus2" }
variable "prefix"   { default = "lhagent" }

resource "azurerm_resource_group" "rg" {
  name     = "${var.prefix}-rg"
  location = var.location
}

# Hierarchical namespace on = ADLS Gen2.
resource "azurerm_storage_account" "lake" {
  name                     = "${var.prefix}lake"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true
  min_tls_version          = "TLS1_2"
}

resource "azurerm_storage_container" "raw" {
  name                  = "raw"
  storage_account_name  = azurerm_storage_account.lake.name
  container_access_type = "private"
}

# Premium SKU is required for Unity Catalog / lineage features.
resource "azurerm_databricks_workspace" "dbx" {
  name                = "${var.prefix}-dbx"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "premium"
}

output "raw_path" {
  value = "abfss://raw@${azurerm_storage_account.lake.name}.dfs.core.windows.net/"
}

output "workspace_url" {
  value = azurerm_databricks_workspace.dbx.workspace_url
}
