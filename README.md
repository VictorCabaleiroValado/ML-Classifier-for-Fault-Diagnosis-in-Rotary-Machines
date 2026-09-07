# Machine Learning for Rotary Machine Fault Diagnosis

**Vibration analysis · Feature engineering · Reproducible evaluation**

An academic Python workflow for fault detection and classification from rotary-machine vibration measurements. Time-domain statistics and FFT features feed a shared, validated evaluation pipeline.

[Project report](CAPSTONE%20PROJECT.pdf) · [Data provenance](docs/DATA_PROVENANCE.md) · [Validation results](results/validated/README.md)

## Data status

| Speed | Time features | Frequency features | Recommended use |
| :--- | :--- | :--- | :--- |
| 25 RPM | Rebuilt from 975 original measurements | Rebuilt from the same measurements | Default exploration, with filename-based labels |
| 50 RPM | Historical table | Historical table | Explicit opt-in; original measurements unavailable |
| 75 RPM | Restored from the owner's local CSV | Historical table | Explicit opt-in; original measurements unavailable |

The four historical 50/75 RPM tables require `--allow-unverified-data`. Their numeric structure is valid, but their labels and feature semantics cannot be verified without raw measurements. They are not suitable for validated performance claims. The historical root-level `*_RPM_Results.txt` files predate these fixes and are superseded for current evaluation.

## Quick start

Use Python **3.9–3.12**. The exact tested dependency set is provided in `requirements-lock.txt`.

```bash
git clone https://github.com/VictorCabaleiroValado/ML-Classifier-for-Fault-Diagnosis-in-Rotary-Machines.git
cd ML-Classifier-for-Fault-Diagnosis-in-Rotary-Machines
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

```bash
python -m pip install -r requirements-lock.txt
python fault_diagnosis.py evaluate --rpm 25 --domain frequency --model tree
```

For the interactive menu:

```bash
python ML_Classifer_Code_with_Menu.py
```

Select **1 → 1 → 3** for 25 RPM, frequency features and a decision tree. Invalid inputs are handled without a traceback. The original filename spelling is retained for compatibility.

### Save an evaluation

```bash
python fault_diagnosis.py evaluate --rpm 25 --domain time --model logistic \
  --seed 42 --output results/local/time_25_logistic.json
```

Output files are created exclusively: an existing file will not be silently replaced. Relative data paths can be supplied with `--data-dir`; default bundled data resolves relative to the code, so the command also works from another directory.

## Evaluation design

- Predictors use an explicit whitelist of **72 time** or **63 frequency** features. Targets, saved CSV indexes and source metadata never enter the model.
- An **80/20 split stratified by fault category** uses the same rows for both tasks. Every class must appear in both partitions.
- Logistic regression, SVM, MLP and the legacy linear baseline use a `StandardScaler` fitted **only on training rows** through a scikit-learn pipeline.
- A separate estimator is fitted for each target. Randomized estimators and splits have a configurable fixed seed.
- Classifiers report per-class precision, recall, F1, macro/weighted averages and confusion matrices. The linear baseline reports MSE and R²; numeric category codes do not have a meaningful continuous ordering.
- JSON reports include the data SHA-256, feature count, split indices, software versions, provenance and any estimator warnings.
- If a custom table includes authentic acquisition-run IDs, use `--group-column group_id` to keep each run in one partition. The supplied tables do not establish acquisition-run independence; row-level holdout results remain exploratory.

Available models: `tree`, `logistic`, `svm`, `mlp`, `linear`.

## Regenerate features from original measurements

Raw measurements are not included. Each dataset requires a manifest with **explicit labels**:

```csv
filename,fault_detected,fault_category
healthy_trial.csv,0,1
bearing_trial.csv,1,2
```

An optional `group_id` column preserves real acquisition-run IDs. Never derive them from arbitrary row blocks. Paths are relative to `--raw-dir`. `fault_detected` is 0 for healthy and 1 for a fault; `fault_category` is a positive integer, with a consistent detection flag per category.

The 25 RPM manifest and category names are included under [data/manifests](data/manifests). The category IDs in the rebuilt 25 RPM data must not be interpreted using the legacy 50/75 RPM numbering.

For a standard CSV, use the named columns `Tachometer, Motor, B1_Z, B1_Y, B1_X, B2_Z, B2_Y, B2_X, Gearbox`:

```bash
python fault_diagnosis.py extract --rpm 25 --domain time \
  --raw-dir /path/to/measurements --manifest data/manifests/25_rpm.csv \
  --output results/local/time_domain_feature_extraction_25.csv
```

The original 25 RPM files have 18 alternating timestamp/sensor columns and three metadata rows after the header. For those files, add **`--legacy-skip-rows 3`**. This setting was checked against their actual layout. Do not assume it applies to other instruments.

Use `--domain frequency` for FFT features. Default output names match the classifier: `FF_feature_extraction_<RPM>.csv` and `time_domain_feature_extraction_<RPM>.csv`.

`--start` and `--end` select manifest rows (1-based, inclusive) without changing labels. Missing files, nonnumeric samples, nonfinite values, invalid labels and empty selections fail before output is written. Use `--overwrite` explicitly to regenerate an existing output.

Constant signals have zero variance/skewness/kurtosis by documented convention; all other kurtosis values use SciPy's Fisher definition. Frequency features use the magnitude of the unnormalized full FFT, preserving the project's convention.

## Repository guide

| Path | Purpose |
| :--- | :--- |
| `fault_diagnosis.py` | Shared classifier, validation, extraction and CLI |
| `ML_Classifier_Code.py`, `ML_Classifer_Code_with_Menu.py` | Compatibility classifier entry points |
| `Time_Domain_FE*.py`, `Frequency_Domain_FE*.py` | Compatibility extraction entry points; use `--help` |
| `tests/` | Regression, data validation, CLI and extraction tests |
| `data/` | Source manifests, category map and provenance hashes |
| `results/validated/` | Current reproducible execution reports |
| `requirements-lock.txt` | Exact tested dependency versions, including test tools |
| [CAPSTONE PROJECT.pdf](CAPSTONE%20PROJECT.pdf) | Original academic report; not revised by this maintenance work |

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs the test suite on Linux with Python 3.11 and 3.12. See [scikit-learn's guidance on data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) for the rationale behind training-only preprocessing.

**[Explore my professional portfolio](https://victorcabaleirovalado.github.io/)**

Maintained by [Victor Cabaleiro Valado](https://github.com/VictorCabaleiroValado) · [LinkedIn](https://www.linkedin.com/in/victorcabaleirovalado/)
