# Governance / Compliance Notes (Mini MLOps Project)

## 1. Purpose and scope
This repository demonstrates an audit-friendly ML workflow for a binary classification model trained on the Breast Cancer Wisconsin dataset. It implements:
- experiment tracking (MLflow)
- reproducible training (pinned dependencies + fixed seeds)
- model documentation artefacts (reports + confusion matrix)
- model lifecycle controls (registration + staging promotion)
- readiness for controlled serving (FastAPI in Day 3)

**Intended use:** educational / portfolio demonstration of MLOps controls and traceability.  
**Not intended for:** clinical diagnosis, medical decision support, or any real-world patient-facing use.

---

## 2. System overview
### Components
- **Trainer (`trainer/train.py`)**: trains model and logs runs to MLflow (params, metrics, artefacts, model).
- **MLflow Tracking Store (`./mlruns`)**: file-based experiment store for runs and artefacts (local development).
- **Model Registry (MLflow UI)**: registered model `breast_cancer_classifier`, Version 1, promoted to **Staging**.

### Key control objective
Ensure every prediction can be traced to a specific model version, which can be traced back to a specific training run, dataset fingerprint, and training configuration.

---

## 3. Data governance
### Dataset
- Source: Breast Cancer Wisconsin dataset (exported to CSV)
- Local file: `data/breast_cancer.csv`

### Data integrity and versioning control
- A SHA256 fingerprint of the dataset is computed in training and logged to MLflow as tag `data_sha256`.
- This provides evidence of the exact dataset used for each run. If the dataset changes, the hash changes.

**Control:** Training runs must log `data_sha256` and dataset row/feature counts in `run_manifest.json`.

---

## 4. Reproducibility controls
### Environment control
- Training dependencies are pinned in `trainer/requirements.txt`.
- Training runs use a fixed `random_state` and a deterministic `train_test_split` with `stratify=y`.

**Control:** Any change to dependencies or training hyperparameters requires a new run and a new model version.

### Training configuration (logged to MLflow)
Logged parameters include:
- `model_type` (Logistic Regression pipeline)
- `C`, `max_iter`
- `test_size`, `random_state`

---

## 5. Model accountability (evaluation evidence)
### Metrics (logged per run)
At minimum:
- accuracy, precision, recall, f1, roc_auc

### Evidence artifacts (logged per run)
- `confusion_matrix.png`
- `classification_report.json`
- `classification_report.txt`
- `run_manifest.json` (run metadata summary: run_id, data hash, dataset shape, metrics)

**Control:** A model cannot be promoted to Staging/Production unless evaluation artifacts exist for the originating run.

---

## 6. Model interface contract (schema)
The model is logged with an MLflow **signature** and `input_example.json`:
- Inputs: 30 numeric features (double)
- Output: model returns P(class=1) (served as probability_positive_class); API also returns a thresholded decision (prediction) and prediction_label.

**Control:** Serving components must validate request schema and feature names before inference (implemented in Day 3 API).

---

## 7. Change management and promotion workflow
### Registry naming
- Registered model name: `breast_cancer_classifier`

### Promotion control
- New training runs produce candidate models.
- Candidate models are registered as a new version.
- Promotion to **Staging** is a controlled step (manual approval via MLflow UI for this mini-project).
- Production promotion (future) is reserved for versions that pass agreed thresholds and review.

**Current state**
- `breast_cancer_classifier` Version 1 is in **Staging**.

**Control:** Serving must reference only Staging/Production (e.g., `models:/breast_cancer_classifier/Staging` or `models:/breast_cancer_classifier@staging`).

---

## 8. Traceability / audit trail
For any served prediction, the following lineage must be obtainable:
1. Served model identifier (name + version or alias/stage)
2. MLflow run_id for that model version (source run)
3. Run artifacts: dataset hash, parameters, metrics, reports

**Evidence location**
- Runs and artifacts: MLflow tracking store (local `./mlruns` in this phase)
- Registry: MLflow Registered Models UI

---

## 8.1 Serving governance (FastAPI)
### Serving target (current)
- Model URI: `models:/breast_cancer_classifier/Staging`
- API uses the registry stage to avoid hardcoding run IDs and to enable rollback via promotion.
- Swagger UI available at `/docs`

### Endpoints (contract)
- `GET /health` → returns service status and whether a model is loaded.
- `GET /model-info` → returns model_name, model_stage, model_version, source_run_id, expected_feature_count, threshold, and positive_class.
- `POST /predict` → validates inputs and returns a traceable prediction.

### Input validation control
The API enforces an exact feature set (30 features). If features are missing or extra, it returns HTTP 422 with `missing` and `extra` lists.

### Probability semantics
The API returns:
- `positive_class: "benign"`
- `probability_positive_class`: `P(class=1)` from `predict_proba(...)[ :, 1 ]`

This prevents misinterpretation of the score (e.g., assuming it is “probability of cancer”).
---

## 9. Logging and monitoring (planned operational control)
### Inference logging
API will log, per request:
- timestamp
- request_id (uuid)
- model name + version (or alias/stage)
- prediction + probability
- validation outcome / errors (if any)

### Monitoring plan (portfolio-level)
- Data drift check (feature summary statistics vs training baseline) on a periodic schedule (weekly/monthly).
- Performance re-evaluation when model is retrained or when drift threshold is exceeded.
- Promotion to Production only after review of drift/performance evidence.

---

## 10. Risk notes and limitations
- Dataset is a public benchmark dataset and does not represent clinical populations; results are not generalizable.
- Model is not calibrated for real diagnostic thresholds or clinical risk management.
- This project demonstrates governance mechanics (lineage, tracking, lifecycle control), not medical safety.

---

## 11. Roles and responsibilities (simplified)
- Owner: Marie Levesque
- Reviewer/Approver (Staging/Production): N/A - portfolio project
- Frequency of review: on each new model version

---

## 12. References (internal)
- Training script: `trainer/train.py`
- Dataset: `data/breast_cancer.csv`
- Trainer deps: `trainer/requirements.txt`
- MLflow UI: `http://127.0.0.1:5001`