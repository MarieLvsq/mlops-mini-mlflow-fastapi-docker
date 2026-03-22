# Controls Mapping (IT Risk / GRC)

| Control Objective | Implementation | Evidence |
|---|---|---|
| Traceability of decisions | `/model-info` exposes model_version + source_run_id; response has request_id | Swagger screenshots; MLflow registry; API output |
| Reproducible training | pinned deps + seeded split | `trainer/requirements.txt`, MLflow params |
| Data provenance | dataset fingerprint `data_sha256` | MLflow tags + `run_manifest.json` |
| Change control | registry versioning + Staging promotion gate | MLflow Model Registry history |
| Input contract enforcement | signature + exact feature-set validation | MLflow signature; API 422 behavior |
| Operational readiness | docker compose runbook; health endpoint | README quickstart; `/health` |
