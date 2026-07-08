import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import glob

# Paths to directories
path_25 = "Fault data split 25"  
path_50 = "Fault data split 50"  
path_75 = "Fault data split 75"  

# Load CSVs only from Fault data split 75
files_25 = glob.glob(path_25 + "/*.csv")  
files_50 = glob.glob(path_50 + "/*.csv")  
files_75 = glob.glob(path_75 + "/*.csv")  

array_dataframes_25 = []  
array_dataframes_50 = []  
array_dataframes_75 = []  

# Load CSVs into dataframes for split 75 only
for filename in files_25:  
    df = pd.read_csv(filename, index_col=None, low_memory=False)
    array_dataframes_25.append(df)

for filename in files_50:  
    df = pd.read_csv(filename, index_col=None, low_memory=False)
    array_dataframes_50.append(df)

for filename in files_75:  
    df = pd.read_csv(filename, index_col=None, low_memory=False)
    array_dataframes_75.append(df)

# Clean DataFrame function
#def clean_dataframe(df):
    # Remove rows where 'Interval' or 'Channel name' appear
    # df = df[~df.iloc[:, 0].str.contains("Interval|Channel name|Unit", na=False)]
    # Reset index after filtering
    # df.reset_index(drop=True, inplace=True)
    # return df

# Clean only 75 split data
# for i in range(len(array_dataframes_25)):  
#     array_dataframes_25[i] = clean_dataframe(array_dataframes_25[i])

# for i in range(len(array_dataframes_50)):  
#     array_dataframes_50[i] = clean_dataframe(array_dataframes_50[i])

# for i in range(len(array_dataframes_75)):  
#    array_dataframes_75[i] = clean_dataframe(array_dataframes_75[i])

# Drop unnecessary intervals
def drop_intervals(arr):
    for data in arr:
        data.drop(0, inplace=True)  # Drop first row
        data.drop(data.columns[[2, 4, 6, 8, 10, 12, 14, 16]], axis=1, inplace=True)  # Drop specific columns


drop_intervals(array_dataframes_25)  
drop_intervals(array_dataframes_50)  
drop_intervals(array_dataframes_75)  

# Time-domain feature columns
COLUMN_NAMES = ["max_Tachometer", "mean_Tachometer", "var_Tachometer", "std_Tachometer", "rms_Tachometer", "kurtosis_Tachometer", "skew_Tachometer", "ptp_Tachometer",
                "max_Motor", "mean_Motor", "var_Motor", "std_Motor", "rms_Motor", "kurtosis_Motor", "skew_Motor", "ptp_Motor",
                "max_B1_Z", "mean_B1_Z", "var_B1_Z", "std_B1_Z", "rms_B1_Z", "kurtosis_B1_Z", "skew_B1_Z", "ptp_B1_Z",
                "max_B1_Y", "mean_B1_Y", "var_B1_Y", "std_B1_Y", "rms_B1_Y", "kurtosis_B1_Y", "skew_B1_Y", "ptp_B1_Y",
                "max_B1_X", "mean_B1_X", "var_B1_X", "std_B1_X", "rms_B1_X", "kurtosis_B1_X", "skew_B1_X", "ptp_B1_X",
                "max_B2_Z", "mean_B2_Z", "var_B2_Z", "std_B2_Z", "rms_B2_Z", "kurtosis_B2_Z", "skew_B2_Z", "ptp_B2_Z",
                "max_B2_Y", "mean_B2_Y", "var_B2_Y", "std_B2_Y", "rms_B2_Y", "kurtosis_B2_Y", "skew_B2_Y", "ptp_B2_Y",
                "max_B2_X", "mean_B2_X", "var_B2_X", "std_B2_X", "rms_B2_X", "kurtosis_B2_X", "skew_B2_X", "ptp_B2_X",
                "max_Gearbox", "mean_Gearbox", "var_Gearbox", "std_Gearbox", "rms_Gearbox", "kurtosis_Gearbox", "skew_Gearbox", "ptp_Gearbox",
                "fault_detected", "fault_category"]

df_extracted = pd.DataFrame(columns=COLUMN_NAMES)

# Time-domain feature extraction function with numeric conversion
def time_domain_feature_extraction(df, df_of_df, fault_detected, fault_category):
    col = 1
    df_time_vals = []
    for i in range(9):  # Iterate through the 9 sensor columns
        X = df.iloc[:, col].dropna()
        col += 1

        # Convert all values to numeric, invalid parsing will be set to NaN
        X = pd.to_numeric(X, errors='coerce')

        # Drop NaN values that result from invalid strings
        X = X.dropna()

        # Calculate time-domain features only if data exists
        if not X.empty:
            df_time_vals.append(np.max(X))                           # Max
            df_time_vals.append(np.mean(X))                          # Mean
            df_time_vals.append(np.var(X))                           # Variance
            df_time_vals.append(np.std(X))                           # Standard Deviation
            df_time_vals.append(np.sqrt(np.mean(X ** 2)))            # RMS (Root Mean Square)
            df_time_vals.append(kurtosis(X, nan_policy='omit'))      # Kurtosis
            df_time_vals.append(skew(X, nan_policy='omit'))          # Skewness
            df_time_vals.append(np.ptp(X))                           # Peak-to-Peak
        else:
            # Fill with NaN if column is empty after conversion
            df_time_vals.extend([np.nan]*8)

    df_time_vals.append(fault_detected)
    df_time_vals.append(fault_category)

    # Add new row to the DataFrame
    df_of_df.loc[len(df_of_df)] = df_time_vals
    
    return df_of_df

# Iterate and extract time-domain features only for split 75
def dataframe_to_csv(df):
    i = 1
    category = 1
    detected = 1
    for df in array_dataframes_75:  
        if 875 < i <= 900:
            detected = 0
        else:
            detected = 1

        df_extracted = time_domain_feature_extraction(array_dataframes_75[i-1], df_extracted, detected, category)  # Changed to array_dataframes_75

        if i % 25 == 0:
            category += 1

        i += 1

    return df_extracted

# Print extracted features
feature_extraction_25 = dataframe_to_csv(array_dataframes_25)
feature_extraction_50 = dataframe_to_csv(array_dataframes_50)
feature_extraction_75 = dataframe_to_csv(array_dataframes_75)

# make into csv file
df_extracted.to_csv('time_domain_feature_extraction_25.csv', na_rep='NULL', header=True)
print("Time-domain features saved to 'time_domain_feature_extraction_25.csv'")
df_extracted.to_csv('time_domain_feature_extraction_50.csv', na_rep='NULL', header=True)
print("Time-domain features saved to 'time_domain_feature_extraction_50.csv'")
df_extracted.to_csv('time_domain_feature_extraction_75.csv', na_rep='NULL', header=True)
print("Time-domain features saved to 'time_domain_feature_extraction_75.csv'")
