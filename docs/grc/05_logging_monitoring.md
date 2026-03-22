# Logging & Monitoring

## Current
- Training-time logging: params/metrics/artifacts in MLflow
- Serving-time traceability: request_id and registry metadata; `/model-info`

## Planned (Day 5+)
- Inference JSONL logging (timestamp, request_id, model_version, source_run_id, prediction, probability, threshold)
- Upload inference logs to MLflow as artifacts
- Drift checks (feature stats vs training baseline)
