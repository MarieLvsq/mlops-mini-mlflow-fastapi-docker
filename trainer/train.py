import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_confusion_matrix_png(cm: np.ndarray, out_path: str, title: str) -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/breast_cancer.csv")
    parser.add_argument("--experiment-name", default="breast-cancer-baseline")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at: {data_path}")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in the dataset CSV.")

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    data_hash = sha256_file(data_path)

    mlflow.set_experiment(args.experiment_name)

    # Put temp artifacts in a local folder so MLflow can log them
    artifacts_dir = Path("artifacts/day1_run")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        # ---- tags (traceability)
        mlflow.set_tag("data_sha256", data_hash)
        mlflow.set_tag("purpose", "baseline_training")
        mlflow.set_tag("problem_type", "binary_classification")

        # ---- params (reproducibility)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=args.C, max_iter=args.max_iter, solver="lbfgs")),
            ]
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        mlflow.log_metrics(metrics)

        # Artifacts
        cm = confusion_matrix(y_test, y_pred)
        cm_path = artifacts_dir / "confusion_matrix.png"
        save_confusion_matrix_png(cm, str(cm_path), "Confusion Matrix (Test)")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = artifacts_dir / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        # Also store a compact text version (reviewers love this)
        report_txt_path = artifacts_dir / "classification_report.txt"
        report_txt_path.write_text(classification_report(y_test, y_pred))

        # Save a small “run manifest” for audit narratives
        manifest = {
            "run_id": run.info.run_id,
            "experiment": args.experiment_name,
            "data_path": data_path,
            "data_sha256": data_hash,
            "n_rows": int(df.shape[0]),
            "n_features": int(X.shape[1]),
            "metrics": metrics,
        }
        manifest_path = artifacts_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(report_txt_path))
        mlflow.log_artifact(str(manifest_path))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("✅ MLflow run logged.")
        print("Run ID:", run.info.run_id)
        print("Data SHA256:", data_hash)
        print("Metrics:", metrics)


if __name__ == "__main__":
    main()
