# Incident Response (Mini Project)

## Triggers
- Wrong model served
- Schema mismatch errors spike
- Unexpected score distributions

## Actions
1) Identify active model via `/model-info`
2) Roll back by promoting previous registry version to Staging
3) Reproduce the run using MLflow run_id and artifacts
