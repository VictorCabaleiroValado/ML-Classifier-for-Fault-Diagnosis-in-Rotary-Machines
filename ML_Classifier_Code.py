import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Reads files from Frequency Domain feature extraction CSV file
df_FF_25 = pd.read_csv("FF_feature_extraction_25.csv")
df_FF_50 = pd.read_csv("FF_feature_extraction_50.csv")
df_FF_75 = pd.read_csv("FF_feature_extraction_75.csv")

# Reads files from Time Domain feature extraction CSV file
df_TD_25 = pd.read_csv("time_domain_feature_extraction_25.csv")
df_TD_50 = pd.read_csv("time_domain_feature_extraction_50.csv")
df_TD_75 = pd.read_csv("time_domain_feature_extraction_75.csv")

# Features that will be used for fault detection by machine learning model
features_FF = ["max_Tachometer", "mean_Tachometer", "var_Tachometer", "std_Tachometer", "sp_Tachometer", "kurtosis_Tachometer", "skew_Tachometer",
                "max_Motor", "mean_Motor", "var_Motor", "std_Motor", "sp_Motor", "kurtosis_Motor", "skew_Motor",
                "max_B1_Z", "mean_B1_Z", "var_B1_Z", "std_B1_Z", "sp_B1_Z", "kurtosis_B1_Z", "skew_B1_Z",
                "max_B1_Y", "mean_B1_Y", "var_B1_Y", "std_B1_Y", "sp_B1_Y", "kurtosis_B1_Y", "skew_B1_Y",
                "max_B1_X", "mean_B1_X", "var_B1_X", "std_B1_X", "sp_B1_X", "kurtosis_B1_X", "skew_B1_X",
                "max_B2_Z", "mean_B2_Z", "var_B2_Z", "std_B2_Z", "sp_B2_Z", "kurtosis_B2_Z", "skew_B2_Z",
                "max_B2_Y", "mean_B2_Y", "var_B2_Y", "std_B2_Y", "sp_B2_Y", "kurtosis_B2_Y", "skew_B2_Y", 
                "max_B2_X", "mean_B2_X", "var_B2_X", "std_B2_X", "sp_B2_X", "kurtosis_B2_X", "skew_B2_X",
                "max_Gearbox", "mean_Gearbox", "var_Gearbox", "std_Gearbox", "sp_Gearbox", "kurtosis_Gearbox", "skew_Gearbox"]

features_TD = ["max_Tachometer","mean_Tachometer","var_Tachometer","std_Tachometer","rms_Tachometer","kurtosis_Tachometer",
               "skew_Tachometer","ptp_Tachometer","max_Motor","mean_Motor","var_Motor","std_Motor","rms_Motor","kurtosis_Motor",
               "skew_Motor","ptp_Motor","max_B1_Z","mean_B1_Z","var_B1_Z","std_B1_Z","rms_B1_Z","kurtosis_B1_Z","skew_B1_Z",
               "ptp_B1_Z","max_B1_Y","mean_B1_Y","var_B1_Y","std_B1_Y","rms_B1_Y","kurtosis_B1_Y","skew_B1_Y","ptp_B1_Y",
               "max_B1_X","mean_B1_X","var_B1_X","std_B1_X","rms_B1_X","kurtosis_B1_X","skew_B1_X","ptp_B1_X","max_B2_Z",
               "mean_B2_Z","var_B2_Z","std_B2_Z","rms_B2_Z","kurtosis_B2_Z","skew_B2_Z","ptp_B2_Z","max_B2_Y","mean_B2_Y",
               "var_B2_Y","std_B2_Y","rms_B2_Y","kurtosis_B2_Y","skew_B2_Y","ptp_B2_Y","max_B2_X","mean_B2_X","var_B2_X","std_B2_X",
               "rms_B2_X","kurtosis_B2_X","skew_B2_X","ptp_B2_X","max_Gearbox","mean_Gearbox","var_Gearbox","std_Gearbox",
               "rms_Gearbox","kurtosis_Gearbox","skew_Gearbox","ptp_Gearbox","fault_detected","fault_category"]


# Frequency Domain Features DecisionTree Model + Performance Metrics

# Testing 25 rpm with fault detected and fault category
X_FF_25 = df_FF_25[features_FF]
Y_FF_detection_25 = df_FF_25["fault_detected"]
Y_FF_category_25 = df_FF_25["fault_category"]

X_train_detection_25, X_test_detection_25, Y_train_detection_25, Y_test_detection_25 = train_test_split(X_FF_25, Y_FF_detection_25, random_state = 0, test_size=0.2, shuffle = True)
X_train_category_25, X_test_category_25, Y_train_category_25, Y_test_category_25 = train_test_split(X_FF_25, Y_FF_category_25, random_state = 0, test_size=0.2, shuffle = True)


# Initialize Linear Regression Model
linear_reg = LinearRegression()

# Linear Regression 25 RPM
linear_FF_detection_25 = linear_reg.fit(X_train_detection_25, Y_train_detection_25)
linear_y_pred_FF_detection_25 = linear_reg.predict(X_test_detection_25)

linear_FF_category_25 = linear_reg.fit(X_train_category_25, Y_train_category_25)
linear_y_pred_FF_category_25 = linear_reg.predict(X_test_category_25)

# Initialize Logistic Regression Model
logistic_clf = LogisticRegression(max_iter=5000)

# Logistic Regression 25 RPM
logistic_FF_detection_25 = logistic_clf.fit(X_train_detection_25, Y_train_detection_25)
logistic_y_pred_FF_detection_25 = logistic_clf.predict(X_test_detection_25)

logistic_FF_category_25 = logistic_clf.fit(X_train_category_25, Y_train_category_25)
logistic_y_pred_FF_category_25 = logistic_clf.predict(X_test_category_25)

# Decision Tree testing for 25, 50 and 75 rpm using frequency features
dtree = DecisionTreeClassifier()

# Decision Tree 25 RPM
dtree_FF_detection_25 = dtree.fit(X_train_detection_25, Y_train_detection_25)
dtree_y_pred_FF_detection_25 = dtree.predict(X_test_detection_25)
dtree_FF_category_25 = dtree.fit(X_train_category_25, Y_train_category_25)
dtree_y_pred_FF_category_25 = dtree.predict(X_test_category_25)

# SVM testing for 25, 50 and 75 rpm using frequency features
svm_clf = svm.SVC()

# SVM 25 RPM
svm_FF_detection_25 = svm_clf.fit(X_train_detection_25, Y_train_detection_25)
svm_y_pred_FF_detection_25 = svm_clf.predict(X_test_detection_25)
svm_FF_category_25 = svm_clf.fit(X_train_category_25,Y_train_category_25)
svm_y_pred_FF_category_25 = svm_clf.predict(X_test_category_25)

# Neural Network 25 RPM



#########



# Testing 50 rpm with fault detected and fault category
X_FF_50 = df_FF_50[features_FF]
Y_FF_detection_50 = df_FF_50["fault_detected"]
Y_FF_category_50 = df_FF_50["fault_category"]

X_train_detection_50, X_test_detection_50, Y_train_detection_50, Y_test_detection_50 = train_test_split(X_FF_50, Y_FF_detection_50, random_state = 0, test_size=0.2, shuffle = True)
X_train_category_50, X_test_category_50, Y_train_category_50, Y_test_category_50 = train_test_split(X_FF_50, Y_FF_category_50, random_state = 0, test_size=0.2, shuffle = True)

# Linear Regression 50 RPM
linear_FF_detection_50 = linear_reg.fit(X_train_detection_50, Y_train_detection_50)
linear_y_pred_FF_detection_50 = linear_reg.predict(X_test_detection_50)

linear_FF_category_50 = linear_reg.fit(X_train_category_50, Y_train_category_50)
linear_y_pred_FF_category_50 = linear_reg.predict(X_test_category_50)

# Logistic Regression 50 RPM
logistic_FF_detection_50 = logistic_clf.fit(X_train_detection_50, Y_train_detection_50)
logistic_y_pred_FF_detection_50 = logistic_clf.predict(X_test_detection_50)

logistic_FF_category_50 = logistic_clf.fit(X_train_category_50, Y_train_category_50)
logistic_y_pred_FF_category_50 = logistic_clf.predict(X_test_category_50)

# Decision Tree 50 RPM
dtree_FF_detection_50 = dtree.fit(X_train_detection_50, Y_train_detection_50)
dtree_y_pred_FF_detection_50 = dtree.predict(X_test_detection_50)
dtree_FF_category_50 = dtree.fit(X_train_category_50, Y_train_category_50)
dtree_y_pred_FF_category_50 = dtree.predict(X_test_category_50)

# SVM 50 RPM
svm_FF_detection_50 = svm_clf.fit(X_train_detection_50, Y_train_detection_50)
svm_y_pred_FF_detection_50 = svm_clf.predict(X_test_detection_50)
svm_FF_category_50 = svm_clf.fit(X_train_category_50, Y_train_category_50)
svm_y_pred_FF_category_50 = svm_clf.predict(X_test_category_50)

# Neural Network 50 RPM



#########



# Testing 75 rpm with fault detected and fault category
X_FF_75 = df_FF_75[features_FF]
Y_FF_detection_75 = df_FF_75["fault_detected"]
Y_FF_category_75 = df_FF_75["fault_category"]

X_train_detection_75, X_test_detection_75, Y_train_detection_75, Y_test_detection_75 = train_test_split(X_FF_75, Y_FF_detection_75, random_state = 0, test_size=0.2, shuffle = True)
X_train_category_75, X_test_category_75, Y_train_category_75, Y_test_category_75 = train_test_split(X_FF_75, Y_FF_category_75, random_state = 0, test_size=0.2, shuffle = True)

# Linear Regression 75 RPM
linear_FF_detection_75 = linear_reg.fit(X_train_detection_75, Y_train_detection_75)
linear_y_pred_FF_detection_75 = linear_reg.predict(X_test_detection_75)

linear_FF_category_75 = linear_reg.fit(X_train_category_75, Y_train_category_75)
linear_y_pred_FF_category_75 = linear_reg.predict(X_test_category_75)

# Logistic Regression 75 RPM
logistic_FF_detection_75 = logistic_clf.fit(X_train_detection_75, Y_train_detection_75)
logistic_y_pred_FF_detection_75 = logistic_clf.predict(X_test_detection_75)

logistic_FF_category_75 = logistic_clf.fit(X_train_category_75, Y_train_category_75)
logistic_y_pred_FF_category_75 = logistic_clf.predict(X_test_category_75)

# Decision Tree 75 RPM
dtree_FF_detection_75 = dtree.fit(X_train_detection_75, Y_train_detection_75)
dtree_y_pred_FF_detection_75 = dtree.predict(X_test_detection_75)
dtree_FF_category_75 = dtree.fit(X_train_category_75, Y_train_category_75)
dtree_y_pred_FF_category_75 = dtree.predict(X_test_category_75)

# SVM 75 RPM
svm_FF_detection_75 = svm_clf.fit(X_train_detection_75, Y_train_detection_75)
svm_y_pred_FF_detection_75 = svm_clf.predict(X_test_detection_75)
svm_FF_category_75 = svm_clf.fit(X_train_category_75, Y_train_category_75)
svm_y_pred_FF_category_75 = svm_clf.predict(X_test_category_75)

# Neural Network 75 RPM

#########

#######################################################################################################################
#Performance Metrics for FF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score


# 25 RPM Metrics
print("\n\n####################################################################################################")
print("\nPeformance Metrics for 25 RPM Dataset\n")
print("####################################################################################################\n\n")


# Detection of fault for 25 rpm

# Linear Regression
print("\nFault Detection Performance Metrics for 25 RPM Dataset (Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_test_detection_25, linear_y_pred_FF_detection_25))
print("R2 Score:", r2_score(Y_test_detection_25, linear_y_pred_FF_detection_25))

# Logistic Regression
print("\nFault Detection Performance Metrics for 25 RPM Dataset (Logistic Regression):\n")
print(classification_report(Y_test_detection_25, logistic_y_pred_FF_detection_25, zero_division=0))
logistic_cm_FF_25_detection = confusion_matrix(Y_test_detection_25, logistic_y_pred_FF_detection_25)
print("Logistic Regression Confusion matrix: ")
print(logistic_cm_FF_25_detection)

# Decision Tree
print("\nFault Detection Performance Metrics for 25 RPM Dataset:\n")
print(classification_report(Y_test_detection_25,dtree_y_pred_FF_detection_25, zero_division=0))
dtree_cm_FF_25_detection = confusion_matrix(Y_test_detection_25,dtree_y_pred_FF_detection_25)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_FF_25_detection)

# SVM
print(classification_report(Y_test_detection_25, svm_y_pred_FF_detection_25, zero_division=0))
svm_cm_FF_25_detection = confusion_matrix(Y_test_detection_25, svm_y_pred_FF_detection_25)
print("SVM Confusion matrix: ")
print(svm_cm_FF_25_detection)


# Categorization of fault for 25 rpm

# Linear Regression
print("\nFault Categorization Performance Metrics for 25 RPM Dataset (Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_test_category_25, linear_y_pred_FF_category_25))
print("R2 Score:", r2_score(Y_test_category_25, linear_y_pred_FF_category_25))

# Logistic Regression
print("\nFault Categorization Performance Metrics for 25 RPM Dataset (Logistic Regression):")
print(classification_report(Y_test_category_25, logistic_y_pred_FF_category_25, zero_division=0))
logistic_cm_FF_25_category = confusion_matrix(Y_test_category_25, logistic_y_pred_FF_category_25)
print("Logistic Regression Confusion matrix: ")
print(logistic_cm_FF_25_category)

# Decision Tree
print("\nFault Categorization Performance Metrics for 25 RPM Dataset:")
print(classification_report(Y_test_category_25,dtree_y_pred_FF_category_25, zero_division=0))
dtree_cm_FF_25_category = confusion_matrix(Y_test_category_25,dtree_y_pred_FF_category_25)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_FF_25_category)

# SVM
print(classification_report(Y_test_category_25, svm_y_pred_FF_category_25, zero_division=0))
svm_cm_FF_25_category = confusion_matrix(Y_test_category_25, svm_y_pred_FF_category_25)
print("SVM Confusion matrix: ")
print(svm_cm_FF_25_category)



# 50 RPM Metrics
print("\n\n####################################################################################################")
print("\nPeformance Metrics for 50 RPM Dataset\n")
print("####################################################################################################\n\n")


# Detection of fault for 50 rpm

# Linear Regression
print("\nFault Detection Performance Metrics for 50 RPM Dataset (Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_test_detection_50, linear_y_pred_FF_detection_50))
print("R2 Score:", r2_score(Y_test_detection_50, linear_y_pred_FF_detection_50))

# Logistic Regression
print("\nFault Detection Performance Metrics for 50 RPM Dataset (Logistic Regression):\n")
print(classification_report(Y_test_detection_50, logistic_y_pred_FF_detection_50, zero_division=0))
logistic_cm_FF_50_detection = confusion_matrix(Y_test_detection_50, logistic_y_pred_FF_detection_50)
print("Logistic Regression Confusion matrix: ")
print(logistic_cm_FF_50_detection)

# Decision Tree
print("\nFault Detection Performance Metrics for 50 RPM Dataset:\n")
print(classification_report(Y_test_detection_50,dtree_y_pred_FF_detection_50, zero_division=0))
dtree_cm_FF_50_detection = confusion_matrix(Y_test_detection_50,dtree_y_pred_FF_detection_50)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_FF_50_detection)

# SVM
print(classification_report(Y_test_detection_50, svm_y_pred_FF_detection_50, zero_division=0))
svm_cm_FF_50_detection = confusion_matrix(Y_test_detection_50, svm_y_pred_FF_detection_50)
print("SVM Confusion matrix: ")
print(svm_cm_FF_50_detection)

# Categorization of fault for 50 rpm

# Linear Regression
print("\nFault Categorization Performance Metrics for 50 RPM Dataset (Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_test_category_50, linear_y_pred_FF_category_50))
print("R2 Score:", r2_score(Y_test_category_50, linear_y_pred_FF_category_50))

# Logistic Regression
print("\nFault Categorization Performance Metrics for 25 RPM Dataset (Logistic Regression):")
print(classification_report(Y_test_category_50, logistic_y_pred_FF_category_50, zero_division=0))
logistic_cm_FF_25_category = confusion_matrix(Y_test_category_25, logistic_y_pred_FF_category_25)
print("Logistic Regression Confusion matrix: ")
print(logistic_cm_FF_25_category)

# Decision Tree
print("\nFault Categorization Performance Metrics for 50 RPM Dataset:\n")
print(classification_report(Y_test_category_50,dtree_y_pred_FF_category_50, zero_division=0))
dtree_cm_FF_50_category = confusion_matrix(Y_test_category_50,dtree_y_pred_FF_category_50)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_FF_50_category)

# SVM
print(classification_report(Y_test_category_50, svm_y_pred_FF_category_50, zero_division=0))
svm_cm_FF_50_category = confusion_matrix(Y_test_category_50, svm_y_pred_FF_category_50)
print("SVM Confusion matrix: ")
print(svm_cm_FF_50_category)


# 75 RPM Metrics
print("\n\n####################################################################################################")
print("\nPeformance Metrics for 75 RPM Dataset\n")
print("####################################################################################################\n\n")


# Detection of fault for 75 rpm

# Linear Regression
print("\nFault Detection Performance Metrics for 75 RPM Dataset (Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_test_detection_75, linear_y_pred_FF_detection_75))
print("R2 Score:", r2_score(Y_test_detection_75, linear_y_pred_FF_detection_75))

# Logistic Regression
print("\nFault Detection Performance Metrics for 75 RPM Dataset (Logistic Regression):\n")
print(classification_report(Y_test_detection_75, logistic_y_pred_FF_detection_75, zero_division=0))
logistic_cm_FF_75_detection = confusion_matrix(Y_test_detection_75, logistic_y_pred_FF_detection_75)
print("Logistic Regression Confusion matrix: ")
print(logistic_cm_FF_75_detection)

# Decision Tree
print("\nFault Detection Performance Metrics for 75 RPM Dataset:\n")
print(classification_report(Y_test_detection_75,dtree_y_pred_FF_detection_75, zero_division=0))
dtree_cm_FF_75_detection = confusion_matrix(Y_test_detection_75,dtree_y_pred_FF_detection_75)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_FF_75_detection)

# SVM
print(classification_report(Y_test_detection_75, svm_y_pred_FF_detection_75, zero_division=0))
svm_cm_FF_75_detection = confusion_matrix(Y_test_detection_75, svm_y_pred_FF_detection_75)
print("SVM Confusion matrix: ")
print(svm_cm_FF_75_detection)

# Categorization of fault for 75 rpm

# Linear Regression
print("\nFault Categorization Performance Metrics for 75 RPM Dataset (Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_test_category_75, linear_y_pred_FF_category_75))
print("R2 Score:", r2_score(Y_test_category_75, linear_y_pred_FF_category_75))

# Logistic Regression
print("\nFault Categorization Performance Metrics for 75 RPM Dataset (Logistic Regression):")
print(classification_report(Y_test_category_75, logistic_y_pred_FF_category_75, zero_division=0))
logistic_cm_FF_75_category = confusion_matrix(Y_test_category_75, logistic_y_pred_FF_category_75)
print("Logistic Regression Confusion matrix: ")
print(logistic_cm_FF_75_category)

# Decision Tree
print("\nFault Categorization Performance Metrics for 75 RPM Dataset:\n")
print(classification_report(Y_test_category_75,dtree_y_pred_FF_category_75, zero_division=0))
dtree_cm_FF_75_category = confusion_matrix(Y_test_category_75,dtree_y_pred_FF_category_75)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_FF_75_category)

# SVM
print(classification_report(Y_test_category_75, svm_y_pred_FF_category_75, zero_division=0))
svm_cm_FF_75_category = confusion_matrix(Y_test_category_75, svm_y_pred_FF_category_75)
print("SVM Confusion matrix: ")
print(svm_cm_FF_75_category)

#######################################################################################################################

# Time Domain Features DecisionTree Model + Performance Metrics

# Testing 25 rpm with fault detected and fault category
X_TD_25 = df_TD_25[features_TD]
Y_TD_detection_25 = df_TD_25["fault_detected"]
Y_TD_category_25 = df_TD_25["fault_category"]

X_TD_train_detection_25, X_TD_test_detection_25, Y_TD_train_detection_25, Y_TD_test_detection_25 = train_test_split(X_TD_25, Y_TD_detection_25, random_state = 0, test_size=0.2, shuffle = True)
X_TD_train_category_25, X_TD_test_category_25, Y_TD_train_category_25, Y_TD_test_category_25 = train_test_split(X_TD_25, Y_TD_category_25, random_state = 0, test_size=0.2, shuffle = True)

# Linear Regression 25 RPM
linear_TD_detection_25 = linear_reg.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
linear_y_pred_TD_detection_25 = linear_reg.predict(X_TD_test_detection_25)

linear_TD_category_25 = linear_reg.fit(X_TD_train_category_25, Y_TD_train_category_25)
linear_y_pred_TD_category_25 = linear_reg.predict(X_TD_test_category_25)

# Logistic Regression 25 RPM 
logistic_TD_detection_25 = logistic_clf.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
logistic_y_pred_TD_detection_25 = logistic_clf.predict(X_TD_test_detection_25)

logistic_TD_category_25 = logistic_clf.fit(X_TD_train_category_25, Y_TD_train_category_25)
logistic_y_pred_TD_category_25 = logistic_clf.predict(X_TD_test_category_25)

# Decision Tree 25 RPM
dtree_TD_detection_25 = dtree.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
dtree_y_pred_TD_detection_25 = dtree.predict(X_TD_test_detection_25)
dtree_TD_category_25 = dtree.fit(X_TD_train_category_25, Y_train_category_25)
dtree_y_pred_TD_category_25 = dtree.predict(X_TD_test_category_25)

# SVM 25 RPM
svm_TD_detection_25 = svm_clf.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
svm_y_pred_TD_detection_25 = svm_clf.predict(X_TD_test_detection_25)
svm_TD_category_25 = svm_clf.fit(X_TD_train_category_25, Y_train_category_25)
svm_y_pred_TD_category_25 = svm_clf.predict(X_TD_test_category_25)

# Neural Network 25 RPM

#########



# Testing 50 rpm with fault detected and fault category
X_TD_50 = df_TD_50[features_TD]
Y_TD_detection_50 = df_TD_50["fault_detected"]
Y_TD_category_50 = df_TD_50["fault_category"]

X_TD_train_detection_50, X_TD_test_detection_50, Y_TD_train_detection_50, Y_TD_test_detection_50 = train_test_split(X_TD_50, Y_TD_detection_50, random_state = 0, test_size=0.2, shuffle = True)
X_TD_train_category_50, X_TD_test_category_50, Y_TD_train_category_50, Y_TD_test_category_50 = train_test_split(X_TD_50, Y_TD_category_50, random_state = 0, test_size=0.2, shuffle = True)

# Linear Regression 50 RPM
linear_TD_detection_50 = linear_reg.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
linear_y_pred_TD_detection_50 = linear_reg.predict(X_TD_test_detection_50)

linear_TD_category_50 = linear_reg.fit(X_TD_train_category_50, Y_TD_train_category_50)
linear_y_pred_TD_category_50 = linear_reg.predict(X_TD_test_category_50)

# Logistic Regression 50 RPM
logistic_TD_detection_50 = logistic_clf.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
logistic_y_pred_TD_detection_50 = logistic_clf.predict(X_TD_test_detection_50)

logistic_TD_category_50 = logistic_clf.fit(X_TD_train_category_50, Y_TD_train_category_50)
logistic_y_pred_TD_category_50 = logistic_clf.predict(X_TD_test_category_50)

# Decision Tree 50 RPM
dtree_TD_detection_50 = dtree.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
dtree_y_pred_TD_detection_50 = dtree.predict(X_TD_test_detection_50)
dtree_TD_category_50 = dtree.fit(X_TD_train_category_50, Y_TD_train_category_50)
dtree_y_pred_TD_category_50 = dtree.predict(X_TD_test_category_50)

# SVM 50 RPM
svm_TD_detection_50 = svm_clf.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
svm_y_pred_TD_detection_50 = svm_clf.predict(X_TD_test_detection_50)
svm_TD_category_50 = svm_clf.fit(X_TD_train_category_50, Y_TD_train_category_50)
svm_y_pred_TD_category_50 = svm_clf.predict(X_TD_test_category_50)

# Neural Network 50 RPM

#########



# Testing 75 rpm with fault detected and fault category
X_TD_75 = df_TD_75[features_TD]
Y_TD_detection_75 = df_TD_75["fault_detected"]
Y_TD_category_75 = df_TD_75["fault_category"]

X_TD_train_detection_75, X_TD_test_detection_75, Y_TD_train_detection_75, Y_TD_test_detection_75 = train_test_split(X_TD_75, Y_TD_detection_75, random_state = 0, test_size=0.2, shuffle = True)
X_TD_train_category_75, X_TD_test_category_75, Y_TD_train_category_75, Y_TD_test_category_75 = train_test_split(X_TD_75, Y_TD_category_75, random_state = 0, test_size=0.2, shuffle = True)


# Linear Regression 75 RPM
linear_TD_detection_75 = linear_reg.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
linear_y_pred_TD_detection_75 = linear_reg.predict(X_TD_test_detection_75)

linear_TD_category_75 = linear_reg.fit(X_TD_train_category_75, Y_TD_train_category_75)
linear_y_pred_TD_category_75 = linear_reg.predict(X_TD_test_category_75)

# Logistic Regression 75 RPM
logistic_TD_detection_75 = logistic_clf.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
logistic_y_pred_TD_detection_75 = logistic_clf.predict(X_TD_test_detection_75)

logistic_TD_category_75 = logistic_clf.fit(X_TD_train_category_75, Y_TD_train_category_75)
logistic_y_pred_TD_category_75 = logistic_clf.predict(X_TD_test_category_75)

# Decision Tree 75 RPM
dtree_dtree_TD_detection_75 = dtree.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
dtree_y_pred_TD_detection_75 = dtree.predict(X_TD_test_detection_75)
dtree_TD_category_75 = dtree.fit(X_TD_train_category_75, Y_TD_train_category_75)
dtree_y_pred_TD_category_75 = dtree.predict(X_TD_test_category_75)

# SVM 75 RPM
svm_TD_detection_75 = svm_clf.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
svm_y_pred_TD_detection_75 = svm_clf.predict(X_TD_test_detection_75)
svm_TD_category_75 = svm_clf.fit(X_TD_train_category_75, Y_TD_train_category_75)
svm_y_pred_TD_category_75 = svm_clf.predict(X_TD_test_category_75)

# Neural Network 75 RPM

#########


#######################################################################################################################
# Performance Metrics for FF

# 25 RPM Metrics
print("\n\n####################################################################################################")
print("\nPeformance Metrics for 25 RPM Dataset\n")
print("####################################################################################################\n\n")


# Detection of fault for 25 rpm


# Linear Regression
print("\nFault Detection Performance (25 RPM - Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_TD_test_detection_25, linear_y_pred_TD_detection_25))
print("R2 Score:", r2_score(Y_TD_test_detection_25, linear_y_pred_TD_detection_25))

# Logistic regression
print("\nFault Detection Performance (25 RPM - Logistic Regression):\n")
print(classification_report(Y_TD_test_detection_25, logistic_y_pred_TD_detection_25, zero_division=0))
logistic_cm_TD_25_detection = confusion_matrix(Y_TD_test_detection_25, logistic_y_pred_TD_detection_25)
print("Confusion Matrix (Detection): ")
print(logistic_cm_TD_25_detection)

# Decision Tree
print("\nFault Detection Performance Metrics for 25 RPM Dataset:\n")
print(classification_report(Y_TD_test_detection_25,dtree_y_pred_TD_detection_25,zero_division=0))
dtree_cm_TD_25_detection = confusion_matrix(Y_TD_test_detection_25,dtree_y_pred_TD_detection_25)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_TD_25_detection)

# SVM
print(classification_report(Y_TD_test_detection_25, svm_y_pred_TD_detection_25,zero_division=0))
svm_cm_TD_25_detection = confusion_matrix(Y_TD_test_detection_25, svm_y_pred_TD_detection_25)
print("SVM Confusion matrix: ")
print(svm_cm_TD_25_detection)


# Categorization of fault for 25 rpm

# Linear Regression
print("\nFault Categorization Performance (25 RPM - Linear Regression):")
print("Mean Squared Error:", mean_squared_error(Y_TD_test_category_25, linear_y_pred_TD_category_25))
print("R2 Score:", r2_score(Y_TD_test_category_25, linear_y_pred_TD_category_25))

# Logistic regression
print("\nFault Categorization Performance (25 RPM - Logistic Regression):")
print(classification_report(Y_TD_test_category_25, logistic_y_pred_TD_category_25, zero_division=0))
logistic_cm_TD_25_category = confusion_matrix(Y_TD_test_category_25, logistic_y_pred_TD_category_25)
print("Confusion Matrix (Categorization): ")
print(logistic_cm_TD_25_category)

# Decision Tree
print("\nFault Categorization Performance Metrics for 25 RPM Dataset:")
print(classification_report(Y_TD_test_category_25,dtree_y_pred_TD_category_25,zero_division=0))
dtree_cm_TD_25_category = confusion_matrix(Y_TD_test_category_25,dtree_y_pred_TD_category_25)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_TD_25_category)

# SVM
print(classification_report(Y_TD_test_category_25, svm_y_pred_TD_category_25,zero_division=0))
svm_cm_TD_25_category = confusion_matrix(Y_TD_test_category_25, svm_y_pred_TD_category_25)
print("SVM Confusion matrix: ")
print(svm_cm_TD_25_category)


# 50 RPM Metrics
print("\n\n####################################################################################################")
print("\nPeformance Metrics for 50 RPM Dataset\n")
print("####################################################################################################\n\n")


# Detection of fault for 50 rpm

# Linear Regression
print("\nFault Detection Performance (50 RPM - Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_TD_test_detection_50, linear_y_pred_TD_detection_50))
print("R2 Score:", r2_score(Y_TD_test_detection_50, linear_y_pred_TD_detection_50))

# Logistic regression
print("\nFault Detection Performance (50 RPM - Logistic Regression):\n")
print(classification_report(Y_TD_test_detection_50, logistic_y_pred_TD_detection_50, zero_division=0))
logistic_cm_TD_50_detection = confusion_matrix(Y_TD_test_detection_50, logistic_y_pred_TD_detection_50)
print("Confusion Matrix (Detection): ")
print(logistic_cm_TD_50_detection)

# Decision Tree
print("\nFault Detection Performance Metrics for 50 RPM Dataset:\n")
print(classification_report(Y_TD_test_detection_50,dtree_y_pred_TD_detection_50,zero_division=0))
dtree_cm_TD_50_detection = confusion_matrix(Y_TD_test_detection_50,dtree_y_pred_TD_detection_50)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_TD_50_detection)

# SVM
print(classification_report(Y_TD_test_detection_50, svm_y_pred_TD_detection_50,zero_division=0))
svm_cm_TD_50_detection = confusion_matrix(Y_TD_test_detection_50, svm_y_pred_TD_detection_50)
print("SVM Confusion matrix: ")
print(svm_cm_TD_50_detection)



# Categorization of fault for 50 rpm

# Linear Regression
print("\nFault Categorization Performance (50 RPM - Linear Regression):")
print("Mean Squared Error:", mean_squared_error(Y_TD_test_category_50, linear_y_pred_TD_category_50))
print("R2 Score:", r2_score(Y_TD_test_category_50, linear_y_pred_TD_category_50))

# Logistic regression
print("\nFault Categorization Performance (50 RPM - Logistic Regression):")
print(classification_report(Y_TD_test_category_50, logistic_y_pred_TD_category_50, zero_division=0))
logistic_cm_TD_50_category = confusion_matrix(Y_TD_test_category_50, logistic_y_pred_TD_category_50)
print("Confusion Matrix (Categorization): ")
print(logistic_cm_TD_50_category)

# Decision Tree
print("\nFault Categorization Performance Metrics for 50 RPM Dataset:\n")
print(classification_report(Y_TD_test_category_50,dtree_y_pred_TD_category_50,zero_division=0))
dtree_cm_TD_50_category = confusion_matrix(Y_TD_test_category_50,dtree_y_pred_TD_category_50)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_TD_50_category)

# SVM
print(classification_report(Y_TD_test_category_50, svm_y_pred_TD_category_50,zero_division=0))
svm_cm_TD_50_category = confusion_matrix(Y_TD_test_category_50, svm_y_pred_TD_category_50)
print("SVM Confusion matrix: ")
print(svm_cm_TD_50_category)

# 75 RPM Metrics
print("\n\n####################################################################################################")
print("\nPeformance Metrics for 75 RPM Dataset\n")
print("####################################################################################################\n\n")


# Detection of fault for 75 rpm

# Linear Regression
print("\nFault Detection Performance (75 RPM - Linear Regression):\n")
print("Mean Squared Error:", mean_squared_error(Y_TD_test_detection_75, linear_y_pred_TD_detection_75))
print("R2 Score:", r2_score(Y_TD_test_detection_75, linear_y_pred_TD_detection_75))

# Logistic regression
print("\nFault Detection Performance (75 RPM - Logistic Regression):\n")
print(classification_report(Y_TD_test_detection_75, logistic_y_pred_TD_detection_75, zero_division=0))
logistic_cm_TD_75_detection = confusion_matrix(Y_TD_test_detection_75, logistic_y_pred_TD_detection_75)
print("Confusion Matrix (Detection): ")
print(logistic_cm_TD_75_detection)

# Decision Tree
print("\nFault Detection Performance Metrics for 75 RPM Dataset:\n")
print(classification_report(Y_TD_test_detection_75,dtree_y_pred_TD_detection_75,zero_division=0))
dtree_cm_TD_75_detection = confusion_matrix(Y_TD_test_detection_75,dtree_y_pred_TD_detection_75)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_TD_75_detection)

# SVM
print(classification_report(Y_TD_test_detection_75,svm_y_pred_TD_detection_75,zero_division=0))
svm_cm_TD_75_detection = confusion_matrix(Y_TD_test_detection_75,svm_y_pred_TD_detection_75)
print("SVM Confusion matrix: ")
print(svm_cm_TD_75_detection)

# Categorization of fault for 75 rpm

# Linear Regression
print("\nFault Categorization Performance (75 RPM - Linear Regression):")
print("Mean Squared Error:", mean_squared_error(Y_TD_test_category_75, linear_y_pred_TD_category_75))
print("R2 Score:", r2_score(Y_TD_test_category_75, linear_y_pred_TD_category_75))

# Logistic regression
print("\nFault Categorization Performance (75 RPM - Logistic Regression):")
print(classification_report(Y_TD_test_category_75, logistic_y_pred_TD_category_75, zero_division=0))
logistic_cm_TD_75_category = confusion_matrix(Y_TD_test_category_75, logistic_y_pred_TD_category_75)
print("Confusion Matrix (Categorization): ")
print(logistic_cm_TD_75_category)

# Decision Tree
print("\nFault Categorization Performance Metrics for 75 RPM Dataset:\n")
print(classification_report(Y_TD_test_category_75,dtree_y_pred_TD_category_75, zero_division=0))
dtree_cm_TD_75_category = confusion_matrix(Y_TD_test_category_75,dtree_y_pred_TD_category_75)
print("Decision Tree Confusion matrix: ")
print(dtree_cm_TD_75_category)

# SVM
print(classification_report(Y_TD_test_category_75,svm_y_pred_TD_category_75,zero_division=0))
svm_cm_TD_75_category = confusion_matrix(Y_TD_test_category_75,svm_y_pred_TD_category_75)
print("SVM Confusion matrix: ")
print(svm_cm_TD_75_category)

#######################################################################################################################