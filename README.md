# ML Classifier for Fault Diagnosis in Rotary Machines

This project implements a machine learning pipeline for **fault diagnosis in rotary machines** using vibration data.  
It supports **fault detection** and **fault classification** across multiple operating speeds using both **time-domain** and **frequency-domain (FFT)** feature extraction.

The system is designed to be modular and interactive, allowing the user to select:
- Operating speed (RPM)
- Feature extraction method
- Machine learning model
- Diagnosis task

---
## Academic Context

This project was developed as my **Bachelor’s Thesis (TFG)** at  
**Universidad Carlos III de Madrid (UC3M)**.

The thesis received a **final grade of 10/10**, awarded for the quality of the methodology, implementation, and experimental results.

<u>**The full written thesis report is included in this repository under TFG_Alberto_del_Río.pdf in the docs folder.**</u>

## Project Overview

**Input**
- Vibration signal data stored as CSV files
- Data collected at different rotational speeds (25, 50, 75 RPM)

**Processing**
- Time-domain feature extraction
- Frequency-domain feature extraction using FFT

**Models**
- Decision Tree
- Support Vector Machine (SVM)
- Logistic Regression
- Linear Regression
- Neural Network

**Tasks**
- **Fault Detection** (Fault vs No Fault)
- **Fault Classification** (39 fault categories)

**Output**
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrices
- Optional saved result logs

---

## Requirements

- Python 3.x  
- Packages:
  - pandas
  - numpy
  - scipy
  - scikit-learn

All dependencies can be installed automatically using `requirements.txt`.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the project
```bash
python main.py
```

## Data

Place CSV files like:

data/raw/25rpm/*.csv
data/raw/50rpm/*.csv
data/raw/75rpm/*.csv

## Running

### 1) Feature extraction
```bash
python feature_extraction_time.py   # or feature_extraction_freq.py
```

#### Outputs:
```bash
data/features/time_25rpm.csv
data/features/freq_25rpm.csv
...
```

### 2) Train / evaluate models
```bash
python models_menu.py
```

Follow the prompts to choose: RPM: 25 / 50 / 75

Feature extraction: frequency or time

Model: LR / LogReg / DT / SVM / NN

Task: detection or categorization

## Fault label mapping

See docs/fault_codes.md.
