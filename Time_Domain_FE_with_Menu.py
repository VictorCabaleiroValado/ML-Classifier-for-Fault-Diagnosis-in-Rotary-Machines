import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import glob
import sys

# Load CSVs with encoding fallback and progress messages

def load_csvs(files, start=1, end=None):
    files = sorted(files)
    files = files[start - 1:end] if end else files[start - 1:]
    dataframes = []
    for idx, filename in enumerate(files):
        print(f"Loading file {start + idx}: {filename}")
        try:
            df = pd.read_csv(filename, index_col=None, low_memory=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filename, index_col=None, low_memory=False, encoding='ISO-8859-1')
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue
        dataframes.append(df)
    return dataframes

# Drop unnecessary intervals
def drop_intervals(arr):
    for i, data in enumerate(arr):
        print(f"Cleaning DataFrame {i+1}/{len(arr)}")
        data.drop(0, inplace=True)
        data.drop(data.columns[[2, 4, 6, 8, 10, 12, 14, 16]], axis=1, inplace=True)

# Time-domain feature columns
COLUMN_NAMES = [
    f"{stat}_{sensor}" for sensor in ["Tachometer", "Motor", "B1_Z", "B1_Y", "B1_X", "B2_Z", "B2_Y", "B2_X", "Gearbox"]
    for stat in ["max", "mean", "var", "std", "rms", "kurtosis", "skew", "ptp"]
] + ["fault_detected", "fault_category"]

df_extracted = pd.DataFrame(columns=COLUMN_NAMES)

# Time-domain feature extraction function
def time_domain_feature_extraction(df, df_of_df, fault_detected, fault_category):
    col = 1
    df_time_vals = []
    for i in range(9):
        X = df.iloc[:, col].dropna()
        col += 1
        X = pd.to_numeric(X, errors='coerce').dropna()
        if not X.empty:
            df_time_vals.extend([
                np.max(X), np.mean(X), np.var(X), np.std(X),
                np.sqrt(np.mean(X ** 2)),
                kurtosis(X, nan_policy='omit'),
                skew(X, nan_policy='omit'),
                np.ptp(X)
            ])
        else:
            df_time_vals.extend([np.nan]*8)
    df_time_vals.extend([fault_detected, fault_category])
    df_of_df.loc[len(df_of_df)] = df_time_vals
    return df_of_df

# Extraction function with progress
def dataframe_to_csv(dataframes, filename):
    global df_extracted
    df_extracted = pd.DataFrame(columns=COLUMN_NAMES)
    category = 1
    for i, df in enumerate(dataframes):
        print(f"Extracting features from DataFrame {i+1}/{len(dataframes)}")
        detected = 0 if 875 < i <= 900 else 1
        df_extracted = time_domain_feature_extraction(df, df_extracted, detected, category)
        if (i + 1) % 25 == 0:
            category += 1
    df_extracted.to_csv(filename, na_rep='NULL', header=True, index=False)
    print(f"Time-domain features saved to '{filename}'")

# --------- MENU-BASED SELECTION ---------
def menu():
    print("Select dataset for time-domain feature extraction:")
    print("1. Fault data split 25")
    print("2. Fault data split 50")
    print("3. Fault data split 75")
    try:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        start_idx = int(input("Enter the start CSV index (1-based, inclusive): ").strip())
        end_idx = input("Enter the end CSV index (1-based, inclusive, leave empty for all after start): ").strip()
        end_idx = int(end_idx) if end_idx else None
    except Exception as e:
        print(f"Invalid input: {e}")
        return

    if choice == '1':
        files = glob.glob("Fault data split 25/*.csv")
        dataframes = load_csvs(files, start=start_idx, end=end_idx)
        drop_intervals(dataframes)
        dataframe_to_csv(dataframes, 'time_domain_feature_extraction_25.csv')
    elif choice == '2':
        files = glob.glob("Fault data split 50/*.csv")
        dataframes = load_csvs(files, start=start_idx, end=end_idx)
        drop_intervals(dataframes)
        dataframe_to_csv(dataframes, 'time_domain_feature_extraction_50.csv')
    elif choice == '3':
        files = glob.glob("Fault data split 75/*.csv")
        dataframes = load_csvs(files, start=start_idx, end=end_idx)
        drop_intervals(dataframes)
        dataframe_to_csv(dataframes, 'time_domain_feature_extraction_75.csv')
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    menu()
