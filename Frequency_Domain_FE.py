import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.stats import skew, kurtosis
import glob
# import stats


# imports all csv files from 3 different folders (25,50,75 rps)

path_25 = "./Data/Fault_data_split_25"
path_50 = "./Data/Fault_data_split_50"
path_75 = "./Data/Fault_data_split_75"

files_25 = glob.glob(path_25 + "/*.csv")
files_50 = glob.glob(path_50 + "/*.csv")
files_75 = glob.glob(path_75 + "/*.csv")

array_dataframes_25 = []
array_dataframes_50 = []
array_dataframes_75 = []

for filename in files_25:
    df = pd.read_csv(filename, index_col=None, low_memory = False)
    array_dataframes_25.append(df)

"""
for filename in files_50:
    df = pd.read_csv(filename, index_col=None, low_memory=False)
    array_dataframes_50.append(df)


for filename in files_75:
    df = pd.read_csv(filename, index_col=None, low_memory=False)
    array_dataframes_75.append(df)
"""

# drops all copies of the timestamp column in arrays

def drop_intervals(arr):

    for data in arr:
        data.drop(0,inplace = True)
        data.drop(1,inplace = True)
        data.drop(2,inplace = True)

        data.drop(data.columns[[2,4,6,8,10,12,14,16]], axis=1, inplace=True)

drop_intervals(array_dataframes_25)
#drop_intervals(array_dataframes_50)
#drop_intervals(array_dataframes_75)

#begin feature extraction - frequency features
# Fast Fourier Transform

COLUMN_NAMES = ["max_Tachometer", "mean_Tachometer", "var_Tachometer", "std_Tachometer", "sp_Tachometer", "kurtosis_Tachometer", "skew_Tachometer",
                "max_Motor", "mean_Motor", "var_Motor", "std_Motor", "sp_Motor", "kurtosis_Motor", "skew_Motor",
                "max_B1_Z", "mean_B1_Z", "var_B1_Z", "std_B1_Z", "sp_B1_Z", "kurtosis_B1_Z", "skew_B1_Z",
                "max_B1_Y", "mean_B1_Y", "var_B1_Y", "std_B1_Y", "sp_B1_Y", "kurtosis_B1_Y", "skew_B1_Y",
                "max_B1_X", "mean_B1_X", "var_B1_X", "std_B1_X", "sp_B1_X", "kurtosis_B1_X", "skew_B1_X",
                "max_B2_Z", "mean_B2_Z", "var_B2_Z", "std_B2_Z", "sp_B2_Z", "kurtosis_B2_Z", "skew_B2_Z",
                "max_B2_Y", "mean_B2_Y", "var_B2_Y", "std_B2_Y", "sp_B2_Y", "kurtosis_B2_Y", "skew_B2_Y",
                "max_B2_X", "mean_B2_X", "var_B2_X", "std_B2_X", "sp_B2_X", "kurtosis_B2_X", "skew_B2_X",
                "max_Gearbox", "mean_Gearbox", "var_Gearbox", "std_Gearbox", "sp_Gearbox", "kurtosis_Gearbox", "skew_Gearbox", 
                "fault_detected", "fault_category"]

df_extracted = pd.DataFrame(columns=COLUMN_NAMES)


def frequency_feature_extraction(df, df_of_df, fault_detected, fault_category):
    # make dataframe for observations

    col = 1;
    df_freq_vals=[];
    for i in range(9):
        
        X = df.iloc[:, col]
        col+=1

        ft = fft(X)

        # finds the magnitude
        magnitude = abs(ft)

        #find values of max, mean, var, signal power, skew, kurtosis, stdev using magnitude

        df_freq_vals.append(np.max(magnitude))
        df_freq_vals.append(np.mean(magnitude))
        df_freq_vals.append(np.var(magnitude))
        df_freq_vals.append(np.std(magnitude))
        signal_power = np.sum(np.abs(np.power(magnitude,2)))/len(magnitude)
        df_freq_vals.append(signal_power)
        # peak_intensity = len(signal.find_peaks(X)[0])
        # df_freq_vals.append(peak_intensity)
        df_freq_vals.append(skew(magnitude))
        df_freq_vals.append(kurtosis(magnitude))
    
    df_freq_vals.append(fault_detected)
    df_freq_vals.append(fault_category)

    # add new row to dataframe
    df_of_df.loc[len(df_of_df)] = df_freq_vals
        
    
    return df_of_df

#extracts feature info for first dataframe from array_dataframes_25

def dataframe_to_csv(df):
    i = 1
    category = 1
    detected = 1
    for dataframe in df:
        if i > 875 and i <= 900:
            detected = 0
        else:
            detected = 1

        df_extraced = frequency_feature_extraction(df[i-1], df_extracted, detected, category)
        
        if(i%25 == 0):
            category += 1

        i+=1

    #print(df_extracted)
    return df_extracted

feature_extraction_25 = dataframe_to_csv(array_dataframes_25)
feature_extraction_50 = dataframe_to_csv(array_dataframes_50)
feature_extraction_75 = dataframe_to_csv(array_dataframes_75)

# make into csv file
df_extracted.to_csv('feature_extraction_25.csv', na_rep='NULL', header=True)
df_extracted.to_csv('feature_extraction_50.csv', na_rep='NULL', header=True)
df_extracted.to_csv('feature_extraction_75.csv', na_rep='NULL', header=True)
