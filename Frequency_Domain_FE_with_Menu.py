
import pandas as pd
import numpy as np
from scipy.fft import fft
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
        data.drop(1, inplace=True)
        data.drop(2, inplace=True)
        data.drop(data.columns[[2, 4, 6, 8, 10, 12, 14, 16]], axis=1, inplace=True)

# Frequency-domain feature columns
COLUMN_NAMES = [
    f"{stat}_{sensor}" for sensor in ["Tachometer", "Motor", "B1_Z", "B1_Y", "B1_X", "B2_Z", "B2_Y", "B2_X", "Gearbox"]
    for stat in ["max", "mean", "var", "std", "sp", "kurtosis", "skew"]
] + ["fault_detected", "fault_category"]

df_extracted = pd.DataFrame(columns=COLUMN_NAMES)

# Frequency-domain feature extraction function
def frequency_feature_extraction(df, df_of_df, fault_detected, fault_category):
    col = 1
    df_freq_vals = []
    for i in range(9):
        X = df.iloc[:, col].dropna()
        col += 1
        X = pd.to_numeric(X, errors='coerce').dropna()
        if not X.empty:
            ft = fft(X)
            mag = np.abs(ft)
            df_freq_vals.extend([
                np.max(mag), np.mean(mag), np.var(mag), np.std(mag),
                np.mean(mag ** 2),
                kurtosis(mag, nan_policy='omit'),
                skew(mag, nan_policy='omit')
            ])
        else:
            df_freq_vals.extend([np.nan] * 7)
    df_freq_vals.extend([fault_detected, fault_category])
    df_of_df.loc[len(df_of_df)] = df_freq_vals
    return df_of_df

# Extraction function with progress
def dataframe_to_csv(dataframes, filename):
    global df_extracted
    df_extracted = pd.DataFrame(columns=COLUMN_NAMES)
    category = 1
    for i, df in enumerate(dataframes):
        print(f"Extracting features from DataFrame {i+1}/{len(dataframes)}")
        detected = 0 if 875 < i <= 900 else 1
        df_extracted = frequency_feature_extraction(df, df_extracted, detected, category)
        if (i + 1) % 25 == 0:
            category += 1
    df_extracted.to_csv(filename, na_rep='NULL', header=True, index=False)
    print(f"Frequency-domain features saved to '{filename}'")

# --------- MENU-BASED SELECTION ---------
def menu():
    print("Select dataset for frequency-domain feature extraction:")
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
        files = glob.glob("DATA/Fault data split 25/*.csv")
        dataframes = load_csvs(files, start=start_idx, end=end_idx)
        drop_intervals(dataframes)
        dataframe_to_csv(dataframes, 'frequency_domain_feature_extraction_25.csv')
    elif choice == '2':
        files = glob.glob("Fault data split 50/*.csv")
        dataframes = load_csvs(files, start=start_idx, end=end_idx)
        drop_intervals(dataframes)
        dataframe_to_csv(dataframes, 'frequency_domain_feature_extraction_50.csv')
    elif choice == '3':
        files = glob.glob("Fault data split 75/*.csv")
        dataframes = load_csvs(files, start=start_idx, end=end_idx)
        drop_intervals(dataframes)
        dataframe_to_csv(dataframes, 'frequency_domain_feature_extraction_75.csv')
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    menu()
