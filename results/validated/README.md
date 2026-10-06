# Reconstructed-data evaluation

All **30 combinations** (three speeds × two domains × five models, two target estimators each) were rerun on 5 October 2026 with reconstructed original measurements. Every report hash matches its current feature table and every provenance record is `reconstructed_from_raw`. No convergence warnings occurred in this run.

## Time-domain decision trees used by the demo

| Speed | Correct categories | Category accuracy | Macro-F1 | Correct binary detection |
|---|---:|---:|---:|---:|
| 25 RPM | 187/195 | 95.8974% | 0.9569 | 195/195 |
| 50 RPM | 189/195 | 96.9231% | 0.9676 | 195/195 |
| 75 RPM | 188/195 | 96.4103% | 0.9638 | 195/195 |

Each speed has its own model, 780 training rows and 195 held-out rows, stratified by category with seed 42. The demo's 39 examples per speed are a deterministic subset, not the denominator of these accuracy values. All labels are derived from original filenames. This is an exploratory row-level holdout, not independent acquisition/machine validation or calibrated confidence.

Only 25 of 975 recordings per speed are healthy. An always-fault binary classifier would already score 97.44%; use per-class recall and confusion matrices rather than binary accuracy alone.

At 50 RPM, one original recording has 34,044 samples while the other 974 have 64,000; all 75 RPM recordings have 64,000. The shorter record is retained as supplied. Full-recording frequency summaries use an unnormalized FFT and carry a duration-comparability limitation. See [data provenance](../../docs/DATA_PROVENANCE.md).

```bash
python fault_diagnosis.py evaluate --rpm 50 --domain time --model tree --seed 42
python fault_diagnosis.py evaluate --rpm 75 --domain frequency --model logistic --seed 42
```

No `--allow-unverified-data` flag is needed for current tables. Reports include both tasks, matrices, source hashes, split indices and dependency versions. Regression MSE/R² for the linear baseline is not classification accuracy. Historical reports before recovery remain in Git history.

Local validation: 50 Python tests and JavaScript parity for all 117 real demo examples. GitHub Actions checks Python 3.11/3.12 and each browser model; consult the run for the published commit for remote status.
