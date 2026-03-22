# System Overview (GRC View)

## Objective
Demonstrate an audit-friendly ML workflow and controlled model serving lifecycle aligned with IT risk controls:
- traceability (run_id, model version)
- reproducibility (pinned deps, deterministic split)
- controlled promotion (registry stage)
- operational evidence (artifacts, logs)

## Components
- Trainer (`trainer/train.py`): trains + logs params/metrics/artifacts/model to MLflow.
- MLflow Tracking Server (Docker): authoritative store for experiments, artifacts and registry.
- FastAPI (`api/main.py`): serves model from MLflow Registry stage (Staging), validates schema, returns traceable outputs.

## Key Flows
1) Train → MLflow run logged → evidence artifacts produced
2) Register model → version created → promote to Staging
3) API loads `models:/breast_cancer_classifier/Staging`
4) Requests validated against signature → response includes probability semantics + metadata
