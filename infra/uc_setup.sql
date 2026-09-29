-- Run once in the Databricks SQL editor after the workspace exists. Adjust names to match databricks.yml variables.
CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.lakehouse_agent;
CREATE VOLUME IF NOT EXISTS main.lakehouse_agent.raw;
