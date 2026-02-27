import os
import uuid
from typing import Any, Dict, List, Tuple, Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from api.schemas import PredictRequest, PredictResponse

APP_NAME = "breast-cancer-api"

# Registry settings
MODEL_NAME = os.getenv("MODEL_NAME", "breast_cancer_classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")  # use Staging today
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

app = FastAPI(title=APP_NAME, version="0.1.0")

_model = None
_expected_features: List[str] = []
_model_version: Optional[str] = None
_source_run_id: Optional[str] = None
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
POSITIVE_CLASS_LABEL = "benign"  # for sklearn breast cancer dataset: class 1 is typically benign
NEGATIVE_CLASS_LABEL = "malignant"

def load_model_and_schema() -> None:
    global _model, _expected_features, _model_version, _source_run_id

    # 1) Load sklearn model for inference (gives predict_proba)
    _model = mlflow.sklearn.load_model(MODEL_URI)

    # 2) Pull model version + source run for traceability
    client = MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if latest:
        mv = latest[0]
        _model_version = mv.version
        _source_run_id = mv.run_id

    # 3) Load pyfunc wrapper only to read MLflow signature metadata
    pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
    sig = pyfunc_model.metadata.signature
    if sig is None or sig.inputs is None:
        raise RuntimeError("Model signature is missing. Re-log the model with infer_signature (Day 2).")

    _expected_features = [inp.name for inp in sig.inputs.inputs]


@app.on_event("startup")
def startup_event():
    load_model_and_schema()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_uri": MODEL_URI,
        "model_version": _model_version,
        "source_run_id": _source_run_id,
        "expected_feature_count": len(_expected_features),
	"threshold": THRESHOLD,
	"positive_class": POSITIVE_CLASS_LABEL
    }


def validate_and_build_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    # Ensure exact feature set (no missing, no extra)
    incoming = set(features.keys())
    expected = set(_expected_features)

    missing = sorted(list(expected - incoming))
    extra = sorted(list(incoming - expected))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature set mismatch",
                "missing": missing,
                "extra": extra,
            },
        )

    # Preserve order expected by model signature
    row = {name: float(features[name]) for name in _expected_features}
    return pd.DataFrame([row])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = req.request_id or str(uuid.uuid4())
    X = validate_and_build_dataframe(req.features)

    # Predict probability and label
    # proba may be shape (n,) or (n,1) depending on signature; normalize
    proba_val = float(_model.predict_proba(X)[:, 1][0])
    # We return probability for the "positive class" (class=1). In this dataset, class=1 is benign.
    proba_pos = float(_model.predict_proba(X)[:, 1][0])  # P(class=1)
    pred = int(proba_pos >= THRESHOLD)
    label = "benign" if pred == 1 else "malignant"

    return PredictResponse(
        request_id=request_id,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
        positive_class=POSITIVE_CLASS_LABEL,
        probability_positive_class=proba_pos,
        prediction=pred,
        prediction_label=label,
        probability=proba_val,
    )
