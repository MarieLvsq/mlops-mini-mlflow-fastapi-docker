# Mini MLOps Stack: MLflow Tracking + Model Registry + FastAPI Serving (Breast Cancer)

An end-to-end, audit-friendly mini project demonstrating:
- MLflow **experiment tracking** (params, metrics, artifacts)
- MLflow **Model Registry** (versioning + Staging promotion)
- **FastAPI** model serving loading from registry stage (`models:/.../Staging`)
- Basic governance artifacts (dataset fingerprinting, model signature, traceability)

> Portfolio / educational project only — not for clinical use.

## Architecture (local dev)

- **trainer/**: trains a Logistic Regression pipeline and logs runs to MLflow  
- **MLflow**: local tracking store (`./mlruns`) + UI  
- **api/**: FastAPI service loading the **Staging** model from MLflow Registry


## Quickstart

### 0) Prerequisites
- Python 3.9+
- (Optional) Git

### 1) Create + activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip```

### 2) Install dependencies
```bash
python3 -m pip install -r trainer/requirements.txt
python3 -m pip install -r api/requirements.txt```

### 3) Ensure dataset exists
```bash
ls -lh data/breast_cancer.csv```

### 4) Run MLflow UI (port 5001)
Port 5000 may be occupied on macOS, so we use 5001.

```bash
export MLFLOW_TRACKING_URI="file:./mlruns"
python3 -m mlflow ui --host 127.0.0.1 --port 5001 --backend-store-uri file:./mlruns```

Open: http://127.0.0.1:5001

### 5) Train and log an MLflow run
```bash
export MLFLOW_TRACKING_URI="file:./mlruns"
python3 trainer/train.py```

In MLflow UI, confirm the latest run contains:
- params + metrics
- artifacts (confusion matrix + classification reports + run_manifest.json)
- model artifact with signature + input_example.json

### 6) Register and promote the model (MLflow UI)
In MLflow UI:
	1. Open the latest run → Artifacts → model
	2. Click Register model
	3. Name: breast_cancer_classifier
	4. Go to Models → breast_cancer_classifier → Version 1 → set stage to Staging

### 7) Start the FastAPI service (serves Staging model)
```bash
export MLFLOW_TRACKING_URI="file:./mlruns"
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### API endpoints:
- GET /health → service status + model loaded flag
- GET /model-info → model name/stage/version/source run + expected features + threshold + positive class
	• POST /predict → strict input validation + traceable prediction output:
	• positive_class, probability_positive_class
	• prediction, prediction_label
	• request_id, model_name, model_stage

### Example request (generate payload from the CSV)
Generate a valid request body using the first row from the dataset:
```bash
python3 - <<'PY'
import json, pandas as pd
df = pd.read_csv("data/breast_cancer.csv")
row = df.drop(columns=["target"]).iloc[0].to_dict()
payload = {"request_id": "demo-001", "features": {k: float(v) for k, v in row.items()}}
print(json.dumps(payload, indent=2))
PY
```

```bash
python3 - <<'PY' > payload.json
import json, pandas as pd
df = pd.read_csv("data/breast_cancer.csv")
row = df.drop(columns=["target"]).iloc[0].to_dict()
payload = {"request_id": "demo-001", "features": {k: float(v) for k, v in row.items()}}
print(json.dumps(payload))
PY

curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Screenshots

Suggested files:
 • docs/screenshots/Day1_MLflow_Run_Overview_Params_Metrics.png
 • docs/screenshots/Day3_API_Predict_Response_With_Traceability.png


### Governance (audit / compliance notes)

See: docs/governance_notes.md
Covers:
 • dataset fingerprinting (data_sha256)
 • reproducibility controls (pinned deps + seeded split)
 • evaluation evidence artefacts
 • model signature contract + registry promotion gate
 • serving traceability (model version + source run id)

### Notes
	• Portfolio demo: no authentication/authorization by default.
	• Not intended for clinical diagnosis or decision support.