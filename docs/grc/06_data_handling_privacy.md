# Data Handling & Privacy

## Data used
- Public benchmark dataset (Breast Cancer Wisconsin) exported to CSV.

## Data minimization
- API accepts numeric features only; no PII fields.
- No raw input persistence in current implementation.

## Logging policy
- Planned inference logs store metadata + scores, not raw sensitive content.
