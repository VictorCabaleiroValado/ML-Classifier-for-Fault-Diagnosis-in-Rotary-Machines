# Execution validation

The current reports cover **30 combinations**: three speeds × two feature domains × five models, with both detection and categorization evaluated separately (60 fitted target models).

These are **software execution checks and exploratory holdout evaluations**, not an independent scientific benchmark. The data status is embedded in every report:

- **25 RPM:** features reconstructed from original measurements with explicit filename-derived labels.
- **50/75 RPM:** historical tables run with explicit `--allow-unverified-data`. Their performance scores must not be used as validated diagnostic accuracy, because original measurements were unavailable for checking labels and feature semantics.

Each JSON file records a data SHA-256, training and test row indices, dependency versions, seed, task metrics, confusion matrices where applicable, and estimator warnings. The files can be reproduced with the corresponding CLI arguments. For example:

```bash
python fault_diagnosis.py evaluate --rpm 25 --domain time --model tree --seed 42
```

To explicitly explore historical data:

```bash
python fault_diagnosis.py evaluate --rpm 75 --domain time --model tree --seed 42 --allow-unverified-data
```

The holdout uses 780 training and 195 test observations. No original acquisition-run identifiers are available in the bundled tables. Generalization to independent acquisitions is therefore not established.

All local regression tests passed on Python 3.9.6 using `requirements-lock.txt`. GitHub Actions separately checks Linux compatibility on Python 3.11 and 3.12.
