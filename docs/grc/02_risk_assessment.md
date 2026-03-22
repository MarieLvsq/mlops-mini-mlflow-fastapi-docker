# Risk Assessment (Mini MLOps Project)

## Scope
Portfolio demo. Not clinical use. Focus is governance mechanics and auditability.

## Top Risks and Mitigations

### R1 — Serving wrong model (configuration / drift to unapproved version)
- Impact: incorrect decisions; audit failure
- Control: serving pinned to registry stage (`Staging`), model metadata exposed in `/model-info`
- Evidence: MLflow registry shows staged version; API returns model_version/source_run_id

### R2 — Schema mismatch (wrong feature names/order)
- Impact: silent prediction corruption
- Control: MLflow signature + API exact feature-set validation (422 on missing/extra)
- Evidence: signature in MLflow; API validation logic; negative test case

### R3 — Non-reproducible training
- Impact: cannot reproduce model for audit/debug
- Control: pinned dependencies; deterministic seed; dataset hash (`data_sha256`)
- Evidence: requirements files; MLflow run params; `run_manifest.json`

### R4 — Insufficient evidence of performance
- Impact: cannot justify promotion decision
- Control: confusion matrix + classification report as artifacts; minimum metric set logged
- Evidence: MLflow artifacts per run

### R5 — Sensitive data leakage via logs
- Impact: privacy breach
- Control: only numeric features accepted; planned inference logging excludes PII; access control is out-of-scope for portfolio
- Evidence: schema types; logging spec in docs
