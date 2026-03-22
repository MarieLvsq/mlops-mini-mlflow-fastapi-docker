# Change Management

## What triggers a new model version
- code changes affecting training or preprocessing
- dependency changes
- dataset change (new hash)
- hyperparameter changes

## Promotion rule (portfolio)
- Train → log evidence → register model version
- Manual promotion to Staging only after reviewing:
  - metrics
  - confusion matrix
  - classification report
  - signature present
