# Machine Learning for Rotary Machine Fault Diagnosis

**Vibration analysis · Feature engineering · Supervised learning**

An academic Python project for exploring **fault detection** and **fault classification** in rotary machines using vibration measurements at 25, 50 and 75 RPM. It combines time-domain statistics and frequency-domain (FFT) features with an interactive model comparison workflow.

[Project report](CAPSTONE%20PROJECT.pdf) · [Interactive classifier](ML_Classifer_Code_with_Menu.py) · [Dependencies](requirements.txt)

## Workflow

```text
Vibration CSVs → Sensor features → Train/test split → Model fitting → Evaluation
                   Time / FFT                        Detection / Category
```

The menu offers decision trees, logistic regression, support vector machines and a multilayer perceptron. A linear regression option is also included as an exploratory baseline; its MSE and R² outputs should not be interpreted as classification accuracy.

Classification outputs include precision, recall, F1-score and confusion matrices. Existing result logs are included for reference; they are historical outputs, not independently validated benchmarks.

## Get started

Run commands from the repository root. The classifier reads the feature CSVs included in this repository.

```bash
git clone https://github.com/VictorValado/ML-Classifier-for-Fault-Diagnosis-in-Rotary-Machines.git
cd ML-Classifier-for-Fault-Diagnosis-in-Rotary-Machines
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch the interactive classifier:

```bash
python -m pip install -r requirements.txt
python ML_Classifer_Code_with_Menu.py
```

The filename above preserves the repository's original spelling, `Classifer`.

For an initial exploration, select **1 → 1 → 3**: 25 RPM, frequency-domain features, decision tree. The program evaluates detection and categorization, then offers another model run. This is an example workflow, not a claim of validated model performance.

## Repository guide

| File | Purpose |
| :--- | :--- |
| `ML_Classifer_Code_with_Menu.py` | Interactive dataset, feature-domain and model selection |
| `ML_Classifier_Code.py` | Alternative analysis script |
| `Time_Domain_FE_with_Menu.py` | Interactive time-domain feature extraction |
| `Frequency_Domain_FE_with_Menu.py` | Interactive FFT feature extraction |
| `Time_Domain_FE.py`, `Frequency_Domain_FE.py` | Alternative extraction scripts |
| `FF_feature_extraction_*.csv` | Frequency-domain feature tables read by the classifier |
| `time_domain_feature_extraction_*.csv` | Time-domain feature files; see the 75 RPM limitation below |
| `25_RPM_Results.txt`, `50_RPM_Results.txt`, `75_RPM_Results.txt` | Historical result logs |
| [CAPSTONE PROJECT.pdf](CAPSTONE%20PROJECT.pdf) | Included project report |
| `requirements.txt` | NumPy, Pandas, SciPy and scikit-learn dependencies |

## Regenerating features

Raw measurement folders are **not included**. The menu-based extraction scripts expect these locations:

| Dataset | Time-domain input | Frequency-domain input |
| :--- | :--- | :--- |
| 25 RPM | `Fault data split 25/*.csv` | `DATA/Fault data split 25/*.csv` |
| 50 RPM | `Fault data split 50/*.csv` | `Fault data split 50/*.csv` |
| 75 RPM | `Fault data split 75/*.csv` | `Fault data split 75/*.csv` |

```bash
python Time_Domain_FE_with_Menu.py
python Frequency_Domain_FE_with_Menu.py
```

The scripts ask for the dataset and CSV index range. Time-domain extraction writes `time_domain_feature_extraction_<RPM>.csv`. Frequency-domain extraction writes `frequency_domain_feature_extraction_<RPM>.csv`, while the classifier expects `FF_feature_extraction_<RPM>.csv`; align these names before using newly generated frequency features.

Feature extraction assumes a specific source-column layout and assigns labels from file order. Verify the original measurement order and label mapping before regeneration, especially when selecting a subset. Running extraction can overwrite an existing output CSV.

## Current limitations and evaluation notes

- **Time-domain target leakage:** the interactive classifier's `features_TD` list includes `fault_detected` and `fault_category`, which are also prediction targets. Remove both from the predictors and rerun evaluation before relying on time-domain results.
- **Invalid 75 RPM time-domain file:** the committed `time_domain_feature_extraction_75.csv` contains a GitHub rate-limit error message rather than a feature table. This path requires regeneration from the original data.
- **Validation design:** the interactive script uses an 80/20 shuffled split with `random_state=0`. When multiple observations originate from the same measurement run, evaluate with a split grouped by run to check generalization.
- **Reproducibility:** package versions are not pinned and some estimators have no fixed random seed. Results may vary between runs and environments.
- **Scope:** this repository is an academic exploration. The published workflow operates on CSV files; it is not a deployed real-time monitoring service.

## Background

The repository connects signal processing, statistical feature extraction and machine learning for engineering diagnostics. The included [project report](CAPSTONE%20PROJECT.pdf) provides further context.

Maintained by [Victor Cabaleiro Valado](https://github.com/VictorValado) · [LinkedIn](https://www.linkedin.com/in/victor-cabaleiro-valado/)
