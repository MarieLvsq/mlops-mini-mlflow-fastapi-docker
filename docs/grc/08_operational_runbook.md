# Operational Runbook

## Start stack
docker compose up -d --build

## URLs
- MLflow: http://127.0.0.1:5001
- API Swagger: http://127.0.0.1:8000/docs

## Populate Docker MLflow
source .venv/bin/activate
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
python3 trainer/train.py

## Restart API after promotion
docker compose restart api
docker compose logs -f api
