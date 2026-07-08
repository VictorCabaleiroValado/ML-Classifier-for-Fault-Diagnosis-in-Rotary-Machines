import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

cont = True;

#begining of menu options - ask what dataset they would like to study
while cont != False:

    print("\nWhat dataset would you like to look at? \n")
    print("Type \"1\" for 25 RPM dataset")
    print("Type \"2\" for 50 RPM dataset")
    print("Type \"3\" for 75 RPM dataset\n")
    data_RPM = input()
    data_RPM = int(data_RPM)

    while data_RPM < 1 or data_RPM > 3:
        print("\nError: input is not 1, 2, or 3")
        print("\nPlease enter a valid input:")
        print("Type \"1\" for 25 RPM dataset")
        print("Type \"2\" for 50 RPM dataset")
        print("Type \"3\" for 75 RPM dataset\n")
        data_RPM = input()
        data_RPM = int(data_RPM)

    print("\nWhat feature extraction type would you like to look at? ")
    print("Type \"1\" for Frequency Domain Features")
    print("Type \"2\" for Time Domain Features\n")
    Extraction_type = input()
    Extraction_type = int(Extraction_type)

    while Extraction_type < 1 or Extraction_type > 2:
        print("\nError: input is not 1 or 2")
        print("\nPlease enter a valid input:")
        print("Type \"1\" for Frequency Domain Features")
        print("Type \"2\" for Time Domain Features\n")
        Extraction_type = input()
        Extraction_type = int(Extraction_type)


    # Reads files from Frequency Domain feature extraction CSV file
    if data_RPM == 1 and Extraction_type == 1:
        df_FF_25 = pd.read_csv("FF_feature_extraction_25.csv")
    if data_RPM == 2 and Extraction_type == 1:
        df_FF_50 = pd.read_csv("FF_feature_extraction_50.csv")
    if data_RPM == 3 and Extraction_type == 1:   
        df_FF_75 = pd.read_csv("FF_feature_extraction_75.csv")

    # Reads files from Time Domain feature extraction CSV file
    if data_RPM == 1 and Extraction_type == 2:
        df_TD_25 = pd.read_csv("time_domain_feature_extraction_25.csv")
    if data_RPM == 2 and Extraction_type == 2:
        df_TD_50 = pd.read_csv("time_domain_feature_extraction_50.csv")
    if data_RPM == 3 and Extraction_type == 2:
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
    
    print("\nWhat model would you like to look at? ")
    print("Type \"1\" for Linear Regression Model")
    print("Type \"2\" for Logistic Regression Model")
    print("Type \"3\" for Decison Tree Model")
    print("Type \"4\" for SVM Model")
    print("Type \"5\" for Neural Network Model\n")
    Model_type = input()
    Model_type = int(Model_type)

    while Model_type < 1 or Model_type > 5:
        print("\nError: input is not 1, 2, 3, 4, or 5")
        print("\nPlease enter a valid input:")
        print("Type \"1\" for Linear Regression Model")
        print("Type \"2\" for Logistic Regression Model")
        print("Type \"3\" for Decison Tree Model")
        print("Type \"4\" for SVM Model")
        print("Type \"5\" for Neural Network Model\n")
        Model_type = input()
        Model_type = int( Model_type)

    # Initialize Linear Regression Model
    linear_reg = LinearRegression()

    # Initialize Logistic Regression Model
    logistic_clf = LogisticRegression(max_iter=10000, solver='saga')

    # Initialize Decision Tree 
    dtree = DecisionTreeClassifier()
        
    # Initialize SVM 
    svm_clf = svm.SVC()

    # Initialize Neural Network
    n_network = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='relu', solver ='adam', max_iter = 500)

##############################################################################################################################


    # Frequency Domain Feature Models for 25 RPM dataset

    # Testing 25 rpm with fault detected and fault category 
    if data_RPM == 1 and Extraction_type == 1:
        X_FF_25 = df_FF_25[features_FF]
        Y_FF_detection_25 = df_FF_25["fault_detected"]
        Y_FF_category_25 = df_FF_25["fault_category"]

        X_train_detection_25, X_test_detection_25, Y_train_detection_25, Y_test_detection_25 = train_test_split(X_FF_25, Y_FF_detection_25, random_state = 0, test_size=0.2, shuffle = True)
        X_train_category_25, X_test_category_25, Y_train_category_25, Y_test_category_25 = train_test_split(X_FF_25, Y_FF_category_25, random_state = 0, test_size=0.2, shuffle = True)

        if Model_type == 1:
            # Linear Regression 25 RPM
            linear_FF_detection_25 = linear_reg.fit(X_train_detection_25, Y_train_detection_25)
            linear_y_pred_FF_detection_25 = linear_reg.predict(X_test_detection_25)
            linear_FF_category_25 = linear_reg.fit(X_train_category_25, Y_train_category_25)
            linear_y_pred_FF_category_25 = linear_reg.predict(X_test_category_25)

        if Model_type == 2:
            # Logistic Regression 25 RPM
            logistic_FF_detection_25 = logistic_clf.fit(X_train_detection_25, Y_train_detection_25)
            logistic_y_pred_FF_detection_25 = logistic_clf.predict(X_test_detection_25)
            logistic_FF_category_25 = logistic_clf.fit(X_train_category_25, Y_train_category_25)
            logistic_y_pred_FF_category_25 = logistic_clf.predict(X_test_category_25)

        if Model_type == 3:
            # Decision Tree 25 RPM
            dtree_FF_detection_25 = dtree.fit(X_train_detection_25, Y_train_detection_25)
            dtree_y_pred_FF_detection_25 = dtree.predict(X_test_detection_25)
            dtree_FF_category_25 = dtree.fit(X_train_category_25, Y_train_category_25)
            dtree_y_pred_FF_category_25 = dtree.predict(X_test_category_25)

        if Model_type == 4:
            # SVM 25 RPM
            svm_FF_detection_25 = svm_clf.fit(X_train_detection_25, Y_train_detection_25)
            svm_y_pred_FF_detection_25 = svm_clf.predict(X_test_detection_25)
            svm_FF_category_25 = svm_clf.fit(X_train_category_25,Y_train_category_25)
            svm_y_pred_FF_category_25 = svm_clf.predict(X_test_category_25)

        if Model_type == 5:
            n_network.fit(X_train_detection_25, Y_train_detection_25)
            neural_network_y_pred_FF_detection_25 = n_network.predict(X_test_detection_25)
            n_network.fit(X_train_category_25, Y_train_category_25)
            neural_network_y_pred_FF_category_25 = n_network.predict(X_test_category_25)

###################################################################################################


    # Frequency Domain Feature Models for 50 RPM dataset
    if data_RPM == 2 and Extraction_type == 1:
        X_FF_50 = df_FF_50[features_FF]
        Y_FF_detection_50 = df_FF_50["fault_detected"]
        Y_FF_category_50 = df_FF_50["fault_category"]

        X_train_detection_50, X_test_detection_50, Y_train_detection_50, Y_test_detection_50 = train_test_split(X_FF_50, Y_FF_detection_50, random_state = 0, test_size=0.2, shuffle = True)
        X_train_category_50, X_test_category_50, Y_train_category_50, Y_test_category_50 = train_test_split(X_FF_50, Y_FF_category_50, random_state = 0, test_size=0.2, shuffle = True)

        if Model_type == 1:
            # Linear Regression 50 RPM
            linear_FF_detection_50 = linear_reg.fit(X_train_detection_50, Y_train_detection_50)
            linear_y_pred_FF_detection_50 = linear_reg.predict(X_test_detection_50)
            linear_FF_category_50 = linear_reg.fit(X_train_category_50, Y_train_category_50)
            linear_y_pred_FF_category_50 = linear_reg.predict(X_test_category_50)

        if Model_type == 2:
            # Logistic Regression 50 RPM
            logistic_FF_detection_50 = logistic_clf.fit(X_train_detection_50, Y_train_detection_50)
            logistic_y_pred_FF_detection_50 = logistic_clf.predict(X_test_detection_50)
            logistic_FF_category_50 = logistic_clf.fit(X_train_category_50, Y_train_category_50)
            logistic_y_pred_FF_category_50 = logistic_clf.predict(X_test_category_50)

        if Model_type == 3:
            # Decision Tree 50 RPM
            dtree_FF_detection_50 = dtree.fit(X_train_detection_50, Y_train_detection_50)
            dtree_y_pred_FF_detection_50 = dtree.predict(X_test_detection_50)
            dtree_FF_category_50 = dtree.fit(X_train_category_50, Y_train_category_50)
            dtree_y_pred_FF_category_50 = dtree.predict(X_test_category_50)

        if Model_type == 4:
            # SVM 50 RPM
            svm_FF_detection_50 = svm_clf.fit(X_train_detection_50, Y_train_detection_50)
            svm_y_pred_FF_detection_50 = svm_clf.predict(X_test_detection_50)
            svm_FF_category_50 = svm_clf.fit(X_train_category_50, Y_train_category_50)
            svm_y_pred_FF_category_50 = svm_clf.predict(X_test_category_50)

        if Model_type == 5:
            # Neural Network 50 RPM
            n_network.fit(X_train_detection_50, Y_train_detection_50)
            neural_network_y_pred_FF_detection_50 = n_network.predict(X_test_detection_50)
            n_network.fit(X_train_category_50, Y_train_category_50)
            neural_network_y_pred_FF_category_50 = n_network.predict(X_test_category_50)

############################################################################################################


    # Frequency Domain Feature Models for 75 RPM dataset
    if data_RPM == 3 and Extraction_type == 1:
        X_FF_75 = df_FF_75[features_FF]
        Y_FF_detection_75 = df_FF_75["fault_detected"]
        Y_FF_category_75 = df_FF_75["fault_category"]

        X_train_detection_75, X_test_detection_75, Y_train_detection_75, Y_test_detection_75 = train_test_split(X_FF_75, Y_FF_detection_75, random_state = 0, test_size=0.2, shuffle = True)
        X_train_category_75, X_test_category_75, Y_train_category_75, Y_test_category_75 = train_test_split(X_FF_75, Y_FF_category_75, random_state = 0, test_size=0.2, shuffle = True)

        if Model_type == 1:
            # Linear Regression 75 RPM
            linear_FF_detection_75 = linear_reg.fit(X_train_detection_75, Y_train_detection_75)
            linear_y_pred_FF_detection_75 = linear_reg.predict(X_test_detection_75)
            linear_FF_category_75 = linear_reg.fit(X_train_category_75, Y_train_category_75)
            linear_y_pred_FF_category_75 = linear_reg.predict(X_test_category_75)

        if Model_type == 2:
            # Logistic Regression 75 RPM
            logistic_FF_detection_75 = logistic_clf.fit(X_train_detection_75, Y_train_detection_75)
            logistic_y_pred_FF_detection_75 = logistic_clf.predict(X_test_detection_75)
            logistic_FF_category_75 = logistic_clf.fit(X_train_category_75, Y_train_category_75)
            logistic_y_pred_FF_category_75 = logistic_clf.predict(X_test_category_75)

        if Model_type == 3:
            # Decision Tree 75 RPM
            dtree_FF_detection_75 = dtree.fit(X_train_detection_75, Y_train_detection_75)
            dtree_y_pred_FF_detection_75 = dtree.predict(X_test_detection_75)
            dtree_FF_category_75 = dtree.fit(X_train_category_75, Y_train_category_75)
            dtree_y_pred_FF_category_75 = dtree.predict(X_test_category_75)

        if Model_type == 4:
            # SVM 75 RPM
            svm_FF_detection_75 = svm_clf.fit(X_train_detection_75, Y_train_detection_75)
            svm_y_pred_FF_detection_75 = svm_clf.predict(X_test_detection_75)
            svm_FF_category_75 = svm_clf.fit(X_train_category_75, Y_train_category_75)
            svm_y_pred_FF_category_75 = svm_clf.predict(X_test_category_75)

        if Model_type == 5:
            # Neural Network 75 RPM
            n_network.fit(X_train_detection_75, Y_train_detection_75)
            neural_network_y_pred_FF_detection_75 = n_network.predict(X_test_detection_75)
            n_network.fit(X_train_category_75, Y_train_category_75)
            neural_network_y_pred_FF_category_75 = n_network.predict(X_test_category_75)

#######################################################################################################################


    #Performance Metrics for Frequency Features
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

    
    if data_RPM == 1 and Extraction_type == 1:

        # 25 RPM Metrics
        print("\n\n####################################################################################################")
        print("\nPeformance Metrics for 25 RPM Dataset with Frequency Domain Features\n")
        print("####################################################################################################\n\n")


        # Detection of fault for 25 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Detection Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_test_detection_25, linear_y_pred_FF_detection_25))
            print("R2 Score:", r2_score(Y_test_detection_25, linear_y_pred_FF_detection_25))

        # Logistic Regression
        if Model_type == 2:
            print("\nFault Detection Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_test_detection_25, logistic_y_pred_FF_detection_25, zero_division=0))
            logistic_cm_FF_25_detection = confusion_matrix(Y_test_detection_25, logistic_y_pred_FF_detection_25)
            print("Logistic Regression Confusion matrix: ")
            print(logistic_cm_FF_25_detection)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Detection Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_test_detection_25,dtree_y_pred_FF_detection_25, zero_division=0))
            dtree_cm_FF_25_detection = confusion_matrix(Y_test_detection_25,dtree_y_pred_FF_detection_25)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_FF_25_detection)

        # SVM
        if Model_type == 4:
            print("\nFault Detection Performance Metrics for SVM Model:\n")
            print(classification_report(Y_test_detection_25, svm_y_pred_FF_detection_25, zero_division=0))
            svm_cm_FF_25_detection = confusion_matrix(Y_test_detection_25, svm_y_pred_FF_detection_25)
            print("SVM Confusion matrix: ")
            print(svm_cm_FF_25_detection)

        # Neural Network
        if Model_type == 5:
            print("\nFault Detection Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_test_detection_25, neural_network_y_pred_FF_detection_25))
            cm_FF_25_detection_NN = confusion_matrix(Y_test_detection_25,neural_network_y_pred_FF_detection_25)
            print("Confusion matrix: ")
            print(cm_FF_25_detection_NN)

        # Categorization of fault for 25 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Categorization Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_test_category_25, linear_y_pred_FF_category_25))
            print("R2 Score:", r2_score(Y_test_category_25, linear_y_pred_FF_category_25))

        # Logistic Regression
        if Model_type == 2:
            print("\nFault Categorization Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_test_category_25, logistic_y_pred_FF_category_25, zero_division=0))
            logistic_cm_FF_25_category = confusion_matrix(Y_test_category_25, logistic_y_pred_FF_category_25)
            print("Logistic Regression Confusion matrix: ")
            print(logistic_cm_FF_25_category)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Categorization Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_test_category_25,dtree_y_pred_FF_category_25, zero_division=0))
            dtree_cm_FF_25_category = confusion_matrix(Y_test_category_25,dtree_y_pred_FF_category_25)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_FF_25_category)

        # SVM
        if Model_type == 4:
            print("\nFault Categorization Performance Metrics for SVM Model:\n")
            print(classification_report(Y_test_category_25, svm_y_pred_FF_category_25, zero_division=0))
            svm_cm_FF_25_category = confusion_matrix(Y_test_category_25, svm_y_pred_FF_category_25)
            print("SVM Confusion matrix: ")
            print(svm_cm_FF_25_category)

        # Neural Network
        if Model_type == 5:
            print("\nFault Categorization Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_test_category_25, neural_network_y_pred_FF_category_25))
            cm_FF_25_category_NN = confusion_matrix(Y_test_category_25,neural_network_y_pred_FF_category_25)
            print("Confusion matrix: ")
            print(cm_FF_25_category_NN)


    if data_RPM == 2 and Extraction_type == 1:
        # 50 RPM Metrics
        print("\n\n####################################################################################################")
        print("\nPeformance Metrics for 50 RPM Dataset with Frequency Domain Features\n")
        print("####################################################################################################\n\n")


        # Detection of fault for 50 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Detection Performance Metrics for Logistic Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_test_detection_50, linear_y_pred_FF_detection_50))
            print("R2 Score:", r2_score(Y_test_detection_50, linear_y_pred_FF_detection_50))

        # Logistic Regression
        if Model_type == 2:
            print("\nFault Detection Performance Metrics for Linear Regression Model:\n")
            print(classification_report(Y_test_detection_50, logistic_y_pred_FF_detection_50, zero_division=0))
            logistic_cm_FF_50_detection = confusion_matrix(Y_test_detection_50, logistic_y_pred_FF_detection_50)
            print("Logistic Regression Confusion matrix: ")
            print(logistic_cm_FF_50_detection)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Detection Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_test_detection_50,dtree_y_pred_FF_detection_50, zero_division=0))
            dtree_cm_FF_50_detection = confusion_matrix(Y_test_detection_50,dtree_y_pred_FF_detection_50)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_FF_50_detection)

        # SVM
        if Model_type == 4:
            print("\nFault Detection Performance Metrics for SVM Model:\n")
            print(classification_report(Y_test_detection_50, svm_y_pred_FF_detection_50, zero_division=0))
            svm_cm_FF_50_detection = confusion_matrix(Y_test_detection_50, svm_y_pred_FF_detection_50)
            print("SVM Confusion matrix: ")
            print(svm_cm_FF_50_detection)

        # Neural Network
        if Model_type == 5:
            print("\nFault Detection Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_test_detection_50, neural_network_y_pred_FF_detection_50))
            cm_FF_50_detection_NN = confusion_matrix(Y_test_detection_50,neural_network_y_pred_FF_detection_50)
            print("Confusion matrix: ")
            print(cm_FF_50_detection_NN)

        # Categorization of fault for 50 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Categorization Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_test_category_50, linear_y_pred_FF_category_50))
            print("R2 Score:", r2_score(Y_test_category_50, linear_y_pred_FF_category_50))

        # Logistic Regression
        if Model_type == 2:
            print("\nFault CategorizationPerformance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_test_category_50, logistic_y_pred_FF_category_50, zero_division=0))
            logistic_cm_FF_50_category = confusion_matrix(Y_test_category_50, logistic_y_pred_FF_category_50)
            print("Logistic Regression Confusion matrix: ")
            print(logistic_cm_FF_50_category)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Categorization Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_test_category_50,dtree_y_pred_FF_category_50, zero_division=0))
            dtree_cm_FF_50_category = confusion_matrix(Y_test_category_50,dtree_y_pred_FF_category_50)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_FF_50_category)

        # SVM
        if Model_type == 4:
            print("\nFault Categorization Performance Metrics for SVM Model:\n")
            print(classification_report(Y_test_category_50, svm_y_pred_FF_category_50, zero_division=0))
            svm_cm_FF_50_category = confusion_matrix(Y_test_category_50, svm_y_pred_FF_category_50)
            print("SVM Confusion matrix: ")
            print(svm_cm_FF_50_category)

        #NN
        if Model_type == 5:
            print("\nFault Categorization Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_test_category_50, neural_network_y_pred_FF_category_50))
            cm_FF_50_category_NN = confusion_matrix(Y_test_category_50,neural_network_y_pred_FF_category_50)
            print("Confusion matrix: ")
            print(cm_FF_50_category_NN)

    if data_RPM == 3 and Extraction_type == 1:
        # 75 RPM Metrics
        print("\n\n####################################################################################################")
        print("\nPeformance Metrics for 75 RPM Dataset with Frequency Domain Features\n")
        print("####################################################################################################\n\n")


        # Detection of fault for 75 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Detection Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_test_detection_75, linear_y_pred_FF_detection_75))
            print("R2 Score:", r2_score(Y_test_detection_75, linear_y_pred_FF_detection_75))

        # Logistic Regression
        if Model_type == 2:
            print("\nFault Detection Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_test_detection_75, logistic_y_pred_FF_detection_75, zero_division=0))
            logistic_cm_FF_75_detection = confusion_matrix(Y_test_detection_75, logistic_y_pred_FF_detection_75)
            print("Logistic Regression Confusion matrix: ")
            print(logistic_cm_FF_75_detection)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Detection Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_test_detection_75,dtree_y_pred_FF_detection_75, zero_division=0))
            dtree_cm_FF_75_detection = confusion_matrix(Y_test_detection_75,dtree_y_pred_FF_detection_75)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_FF_75_detection)

        # SVM
        if Model_type == 4:
            print("\nFault Detection Performance Metrics for SVM Model:\n")
            print(classification_report(Y_test_detection_75, svm_y_pred_FF_detection_75, zero_division=0))
            svm_cm_FF_75_detection = confusion_matrix(Y_test_detection_75, svm_y_pred_FF_detection_75)
            print("SVM Confusion matrix: ")
            print(svm_cm_FF_75_detection)

        #NN
        if Model_type == 5:
            print("\nFault Detection Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_test_detection_75, neural_network_y_pred_FF_detection_75))
            cm_FF_75_detection_NN = confusion_matrix(Y_test_detection_75,neural_network_y_pred_FF_detection_75)
            print("Confusion matrix: ")
            print(cm_FF_75_detection_NN)

        # Categorization of fault for 75 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Categorization Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_test_category_75, linear_y_pred_FF_category_75))
            print("R2 Score:", r2_score(Y_test_category_75, linear_y_pred_FF_category_75))

        # Logistic Regression
        if Model_type == 2:
            print("\nFault Categorization Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_test_category_75, logistic_y_pred_FF_category_75, zero_division=0))
            logistic_cm_FF_75_category = confusion_matrix(Y_test_category_75, logistic_y_pred_FF_category_75)
            print("Logistic Regression Confusion matrix: ")
            print(logistic_cm_FF_75_category)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Categorization Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_test_category_75,dtree_y_pred_FF_category_75, zero_division=0))
            dtree_cm_FF_75_category = confusion_matrix(Y_test_category_75,dtree_y_pred_FF_category_75)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_FF_75_category)

        # SVM
        if Model_type == 4:
            print("\nFault Categorization Performance Metrics for SVM Model:\n")
            print(classification_report(Y_test_category_75, svm_y_pred_FF_category_75, zero_division=0))
            svm_cm_FF_75_category = confusion_matrix(Y_test_category_75, svm_y_pred_FF_category_75)
            print("SVM Confusion matrix: ")
            print(svm_cm_FF_75_category)

        #NN
        if Model_type == 5:
            print("\nFault Categorization Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_test_category_75, neural_network_y_pred_FF_category_75))
            cm_FF_75_category_NN = confusion_matrix(Y_test_category_75,neural_network_y_pred_FF_category_75)
            print("Confusion matrix: ")
            print(cm_FF_75_category_NN)

#######################################################################################################################

    # Time Domain Feature Models 

    # Testing 25 rpm with fault detected and fault category
    if data_RPM == 1 and Extraction_type == 2:
        X_TD_25 = df_TD_25[features_TD]
        Y_TD_detection_25 = df_TD_25["fault_detected"]
        Y_TD_category_25 = df_TD_25["fault_category"]

        X_TD_train_detection_25, X_TD_test_detection_25, Y_TD_train_detection_25, Y_TD_test_detection_25 = train_test_split(X_TD_25, Y_TD_detection_25, random_state = 0, test_size=0.2, shuffle = True)
        X_TD_train_category_25, X_TD_test_category_25, Y_TD_train_category_25, Y_TD_test_category_25 = train_test_split(X_TD_25, Y_TD_category_25, random_state = 0, test_size=0.2, shuffle = True)

        if Model_type == 1:
            # Linear Regression 25 RPM
            linear_TD_detection_25 = linear_reg.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
            linear_y_pred_TD_detection_25 = linear_reg.predict(X_TD_test_detection_25)
            linear_TD_category_25 = linear_reg.fit(X_TD_train_category_25, Y_TD_train_category_25)
            linear_y_pred_TD_category_25 = linear_reg.predict(X_TD_test_category_25)

        if Model_type == 2:
            # Logistic Regression 25 RPM 
            logistic_TD_detection_25 = logistic_clf.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
            logistic_y_pred_TD_detection_25 = logistic_clf.predict(X_TD_test_detection_25)
            logistic_TD_category_25 = logistic_clf.fit(X_TD_train_category_25, Y_TD_train_category_25)
            logistic_y_pred_TD_category_25 = logistic_clf.predict(X_TD_test_category_25)

        if Model_type == 3:
            # Decision Tree 25 RPM
            dtree_TD_detection_25 = dtree.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
            dtree_y_pred_TD_detection_25 = dtree.predict(X_TD_test_detection_25)
            dtree_TD_category_25 = dtree.fit(X_TD_train_category_25, Y_TD_train_category_25)
            dtree_y_pred_TD_category_25 = dtree.predict(X_TD_test_category_25)

        if Model_type == 4:
            # SVM 25 RPM
            svm_TD_detection_25 = svm_clf.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
            svm_y_pred_TD_detection_25 = svm_clf.predict(X_TD_test_detection_25)
            svm_TD_category_25 = svm_clf.fit(X_TD_train_category_25, Y_TD_train_category_25)
            svm_y_pred_TD_category_25 = svm_clf.predict(X_TD_test_category_25)

        if Model_type == 5:
            # Neural Network 25 RPM
            NN_TD_detection_25 = n_network.fit(X_TD_train_detection_25, Y_TD_train_detection_25)
            NN_y_pred_TD_detection_25 = n_network.predict(X_TD_test_detection_25)
            NN_TD_category_25 = n_network.fit(X_TD_train_category_25, Y_TD_train_category_25)
            NN_y_pred_TD_category_25 = n_network.predict(X_TD_test_category_25)

############################################################################################################



    # Testing 50 rpm with fault detected and fault category
    if data_RPM == 2 and Extraction_type == 2:
        X_TD_50 = df_TD_50[features_TD]
        Y_TD_detection_50 = df_TD_50["fault_detected"]
        Y_TD_category_50 = df_TD_50["fault_category"]

        X_TD_train_detection_50, X_TD_test_detection_50, Y_TD_train_detection_50, Y_TD_test_detection_50 = train_test_split(X_TD_50, Y_TD_detection_50, random_state = 0, test_size=0.2, shuffle = True)
        X_TD_train_category_50, X_TD_test_category_50, Y_TD_train_category_50, Y_TD_test_category_50 = train_test_split(X_TD_50, Y_TD_category_50, random_state = 0, test_size=0.2, shuffle = True)

        if Model_type == 1:
            # Linear Regression 50 RPM
            linear_TD_detection_50 = linear_reg.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
            linear_y_pred_TD_detection_50 = linear_reg.predict(X_TD_test_detection_50)
            linear_TD_category_50 = linear_reg.fit(X_TD_train_category_50, Y_TD_train_category_50)
            linear_y_pred_TD_category_50 = linear_reg.predict(X_TD_test_category_50)

        if Model_type == 2:
            # Logistic Regression 50 RPM
            logistic_TD_detection_50 = logistic_clf.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
            logistic_y_pred_TD_detection_50 = logistic_clf.predict(X_TD_test_detection_50)
            logistic_TD_category_50 = logistic_clf.fit(X_TD_train_category_50, Y_TD_train_category_50)
            logistic_y_pred_TD_category_50 = logistic_clf.predict(X_TD_test_category_50)

        if Model_type == 3:
            # Decision Tree 50 RPM
            dtree_TD_detection_50 = dtree.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
            dtree_y_pred_TD_detection_50 = dtree.predict(X_TD_test_detection_50)
            dtree_TD_category_50 = dtree.fit(X_TD_train_category_50, Y_TD_train_category_50)
            dtree_y_pred_TD_category_50 = dtree.predict(X_TD_test_category_50)

        if Model_type == 4:
            # SVM 50 RPM
            svm_TD_detection_50 = svm_clf.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
            svm_y_pred_TD_detection_50 = svm_clf.predict(X_TD_test_detection_50)
            svm_TD_category_50 = svm_clf.fit(X_TD_train_category_50, Y_TD_train_category_50)
            svm_y_pred_TD_category_50 = svm_clf.predict(X_TD_test_category_50)

        if Model_type == 5:
            # Neural Network 50 RPM
            NN_TD_detection_50 = n_network.fit(X_TD_train_detection_50, Y_TD_train_detection_50)
            NN_y_pred_TD_detection_50 = n_network.predict(X_TD_test_detection_50)
            NN_TD_category_50 = n_network.fit(X_TD_train_category_50, Y_TD_train_category_50)
            NN_y_pred_TD_category_50 = n_network.predict(X_TD_test_category_50)

############################################################################################################

    # Testing 75 rpm with fault detected and fault category
    
    if data_RPM == 3 and Extraction_type == 2:
        X_TD_75 = df_TD_75[features_TD]
        Y_TD_detection_75 = df_TD_75["fault_detected"]
        Y_TD_category_75 = df_TD_75["fault_category"]

        X_TD_train_detection_75, X_TD_test_detection_75, Y_TD_train_detection_75, Y_TD_test_detection_75 = train_test_split(X_TD_75, Y_TD_detection_75, random_state = 0, test_size=0.2, shuffle = True)
        X_TD_train_category_75, X_TD_test_category_75, Y_TD_train_category_75, Y_TD_test_category_75 = train_test_split(X_TD_75, Y_TD_category_75, random_state = 0, test_size=0.2, shuffle = True)


        if Model_type == 1:
            # Linear Regression 75 RPM
            linear_TD_detection_75 = linear_reg.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
            linear_y_pred_TD_detection_75 = linear_reg.predict(X_TD_test_detection_75)
            linear_TD_category_75 = linear_reg.fit(X_TD_train_category_75, Y_TD_train_category_75)
            linear_y_pred_TD_category_75 = linear_reg.predict(X_TD_test_category_75)

        if Model_type == 2:
            # Logistic Regression 75 RPM
            logistic_TD_detection_75 = logistic_clf.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
            logistic_y_pred_TD_detection_75 = logistic_clf.predict(X_TD_test_detection_75)
            logistic_TD_category_75 = logistic_clf.fit(X_TD_train_category_75, Y_TD_train_category_75)
            logistic_y_pred_TD_category_75 = logistic_clf.predict(X_TD_test_category_75)

        if Model_type == 3:
            # Decision Tree 75 RPM
            dtree_dtree_TD_detection_75 = dtree.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
            dtree_y_pred_TD_detection_75 = dtree.predict(X_TD_test_detection_75)
            dtree_TD_category_75 = dtree.fit(X_TD_train_category_75, Y_TD_train_category_75)
            dtree_y_pred_TD_category_75 = dtree.predict(X_TD_test_category_75)

        if Model_type == 4:
            # SVM 75 RPM
            svm_TD_detection_75 = svm_clf.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
            svm_y_pred_TD_detection_75 = svm_clf.predict(X_TD_test_detection_75)
            svm_TD_category_75 = svm_clf.fit(X_TD_train_category_75, Y_TD_train_category_75)
            svm_y_pred_TD_category_75 = svm_clf.predict(X_TD_test_category_75)

        if Model_type == 5:
            # Neural Network 75 RPM
            NN_TD_detection_75 = n_network.fit(X_TD_train_detection_75, Y_TD_train_detection_75)
            NN_y_pred_TD_detection_75 = n_network.predict(X_TD_test_detection_75)
            NN_TD_category_75 = n_network.fit(X_TD_train_category_75, Y_TD_train_category_75)
            NN_y_pred_TD_category_75 = n_network.predict(X_TD_test_category_75)

#######################################################################################################################


    # Performance Metrics for Time Domain Features

    if data_RPM == 1 and Extraction_type == 2:

        # 25 RPM Metrics
        print("\n\n####################################################################################################")
        print("\nPeformance Metrics for 25 RPM Dataset with Time Domain Features\n")
        print("####################################################################################################\n\n")

        # Detection of fault for 25 rpm

        # Linear Regression

        if Model_type == 1:
            print("\nFault Detection Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_TD_test_detection_25, linear_y_pred_TD_detection_25))
            print("R2 Score:", r2_score(Y_TD_test_detection_25, linear_y_pred_TD_detection_25))

        # Logistic regression
        if Model_type == 2:
            print("\nFault Detection Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_TD_test_detection_25, logistic_y_pred_TD_detection_25, zero_division=0))
            logistic_cm_TD_25_detection = confusion_matrix(Y_TD_test_detection_25, logistic_y_pred_TD_detection_25)
            print("Confusion Matrix (Detection): ")
            print(logistic_cm_TD_25_detection)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Detection Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_TD_test_detection_25,dtree_y_pred_TD_detection_25,zero_division=0))
            dtree_cm_TD_25_detection = confusion_matrix(Y_TD_test_detection_25,dtree_y_pred_TD_detection_25)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_TD_25_detection)

        # SVM
        if Model_type == 4:
            print("\nFault Detection Performance Metrics for SVM Model:\n")
            print(classification_report(Y_TD_test_detection_25, svm_y_pred_TD_detection_25,zero_division=0))
            svm_cm_TD_25_detection = confusion_matrix(Y_TD_test_detection_25, svm_y_pred_TD_detection_25)
            print("SVM Confusion matrix: ")
            print(svm_cm_TD_25_detection)

        #NN
        if Model_type == 5:
            print("\nFault Detection Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_TD_test_detection_25, NN_y_pred_TD_detection_25))
            cm_TD_25_detection_NN = confusion_matrix(Y_TD_test_detection_25,NN_y_pred_TD_detection_25)
            print("Confusion matrix: ")
            print(cm_TD_25_detection_NN)


        # Categorization of fault for 25 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Categorization Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_TD_test_category_25, linear_y_pred_TD_category_25))
            print("R2 Score:", r2_score(Y_TD_test_category_25, linear_y_pred_TD_category_25))

        # Logistic regression
        if Model_type == 2:
            print("\nFault Categorization Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_TD_test_category_25, logistic_y_pred_TD_category_25, zero_division=0))
            logistic_cm_TD_25_category = confusion_matrix(Y_TD_test_category_25, logistic_y_pred_TD_category_25)
            print("Confusion Matrix (Categorization): ")
            print(logistic_cm_TD_25_category)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Categorization Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_TD_test_category_25,dtree_y_pred_TD_category_25,zero_division=0))
            dtree_cm_TD_25_category = confusion_matrix(Y_TD_test_category_25,dtree_y_pred_TD_category_25)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_TD_25_category)

        # SVM
        if Model_type == 4:
            print("\nFault Categorization Performance Metrics for SVM Model:\n")
            print(classification_report(Y_TD_test_category_25, svm_y_pred_TD_category_25,zero_division=0))
            svm_cm_TD_25_category = confusion_matrix(Y_TD_test_category_25, svm_y_pred_TD_category_25)
            print("SVM Confusion matrix: ")
            print(svm_cm_TD_25_category)

        #NN
        if Model_type == 5:
            print("\nFault Categorization Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_TD_test_category_25, NN_y_pred_TD_category_25))
            cm_FF_25_category_NN = confusion_matrix(Y_TD_test_category_25,NN_y_pred_TD_category_25)
            print("Confusion matrix: ")
            print(cm_FF_25_category_NN)

    if data_RPM == 2 and Extraction_type == 2:
        # 50 RPM Metrics
        print("\n\n####################################################################################################")
        print("\nPeformance Metrics for 50 RPM Dataset with Time Domain Features\n")
        print("####################################################################################################\n\n")

        # Detection of fault for 50 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Detection Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_TD_test_detection_50, linear_y_pred_TD_detection_50))
            print("R2 Score:", r2_score(Y_TD_test_detection_50, linear_y_pred_TD_detection_50))

        # Logistic regression
        if Model_type == 2:
            print("\nFault Detection Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_TD_test_detection_50, logistic_y_pred_TD_detection_50, zero_division=0))
            logistic_cm_TD_50_detection = confusion_matrix(Y_TD_test_detection_50, logistic_y_pred_TD_detection_50)
            print("Confusion Matrix (Detection): ")
            print(logistic_cm_TD_50_detection)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Detection Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_TD_test_detection_50,dtree_y_pred_TD_detection_50,zero_division=0))
            dtree_cm_TD_50_detection = confusion_matrix(Y_TD_test_detection_50,dtree_y_pred_TD_detection_50)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_TD_50_detection)

        # SVM
        if Model_type == 4:
            print("\nFault Detection Performance Metrics for SVM Model:\n")
            print(classification_report(Y_TD_test_detection_50, svm_y_pred_TD_detection_50,zero_division=0))
            svm_cm_TD_50_detection = confusion_matrix(Y_TD_test_detection_50, svm_y_pred_TD_detection_50)
            print("SVM Confusion matrix: ")
            print(svm_cm_TD_50_detection)

        #NN
        if Model_type == 5:
            print("\nFault Detection Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_TD_test_detection_50, NN_y_pred_TD_detection_50))
            cm_TD_50_detection_NN = confusion_matrix(Y_TD_test_detection_50,NN_y_pred_TD_detection_50)
            print("Confusion matrix: ")
            print(cm_TD_50_detection_NN)

        # Categorization of fault for 50 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Categorization Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_TD_test_category_50, linear_y_pred_TD_category_50))
            print("R2 Score:", r2_score(Y_TD_test_category_50, linear_y_pred_TD_category_50))

        # Logistic regression
        if Model_type == 2:
            print("\nFault Categorization Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_TD_test_category_50, logistic_y_pred_TD_category_50, zero_division=0))
            logistic_cm_TD_50_category = confusion_matrix(Y_TD_test_category_50, logistic_y_pred_TD_category_50)
            print("Confusion Matrix (Categorization): ")
            print(logistic_cm_TD_50_category)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Categorization Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_TD_test_category_50,dtree_y_pred_TD_category_50,zero_division=0))
            dtree_cm_TD_50_category = confusion_matrix(Y_TD_test_category_50,dtree_y_pred_TD_category_50)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_TD_50_category)

        # SVM
        if Model_type == 4:
            print("\nFault Categorization Performance Metrics for SVM Model:\n")
            print(classification_report(Y_TD_test_category_50, svm_y_pred_TD_category_50,zero_division=0))
            svm_cm_TD_50_category = confusion_matrix(Y_TD_test_category_50, svm_y_pred_TD_category_50)
            print("SVM Confusion matrix: ")
            print(svm_cm_TD_50_category)

        #NN
        if Model_type == 5:
            print("\nFault Categorization Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_TD_test_category_50, NN_y_pred_TD_category_50))
            cm_FF_50_category_NN = confusion_matrix(Y_TD_test_category_50,NN_y_pred_TD_category_50)
            print("Confusion matrix: ")
            print(cm_FF_50_category_NN)


    if data_RPM == 3 and Extraction_type == 2:
        # 75 RPM Metrics
        print("\n\n####################################################################################################")
        print("\nPeformance Metrics for 75 RPM Dataset with Time Domain Features\n")
        print("####################################################################################################\n\n")

        # Detection of fault for 75 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Detection Performance Metrics for Linear Regression Model:\n")
            print("Mean Squared Error:", mean_squared_error(Y_TD_test_detection_75, linear_y_pred_TD_detection_75))
            print("R2 Score:", r2_score(Y_TD_test_detection_75, linear_y_pred_TD_detection_75))

        # Logistic regression
        if Model_type == 2:
            print("\nFault Detection Performance Metrics for Logistic Regression Model:\n")
            print(classification_report(Y_TD_test_detection_75, logistic_y_pred_TD_detection_75, zero_division=0))
            logistic_cm_TD_75_detection = confusion_matrix(Y_TD_test_detection_75, logistic_y_pred_TD_detection_75)
            print("Confusion Matrix (Detection): ")
            print(logistic_cm_TD_75_detection)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Detection Performance Metrics for Decision Tree Model:\n")
            print(classification_report(Y_TD_test_detection_75,dtree_y_pred_TD_detection_75,zero_division=0))
            dtree_cm_TD_75_detection = confusion_matrix(Y_TD_test_detection_75,dtree_y_pred_TD_detection_75)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_TD_75_detection)

        # SVM
        if Model_type == 4:
            print("\nFault Detection Performance Metrics for SVM Model:\n")
            print(classification_report(Y_TD_test_detection_75,svm_y_pred_TD_detection_75,zero_division=0))
            svm_cm_TD_75_detection = confusion_matrix(Y_TD_test_detection_75,svm_y_pred_TD_detection_75)
            print("SVM Confusion matrix: ")
            print(svm_cm_TD_75_detection)

        #NN
        if Model_type == 5:
            print("\nFault Detection Performance Metrics for Neural Network Model:\n")
            print(classification_report(Y_TD_test_detection_75, NN_y_pred_TD_detection_75))
            cm_TD_75_detection_NN = confusion_matrix(Y_TD_test_detection_75,NN_y_pred_TD_detection_75)
            print("Confusion matrix: ")
            print(cm_TD_75_detection_NN)

        # Categorization of fault for 75 rpm

        # Linear Regression
        if Model_type == 1:
            print("\nFault Categorization Performance Metrics for Linear Regression Model:")
            print("Mean Squared Error:", mean_squared_error(Y_TD_test_category_75, linear_y_pred_TD_category_75))
            print("R2 Score:", r2_score(Y_TD_test_category_75, linear_y_pred_TD_category_75))

        # Logistic regression
        if Model_type == 2:
            print("\nFault Categorization Performance Metrics for Logistic Regression Model:")
            print(classification_report(Y_TD_test_category_75, logistic_y_pred_TD_category_75, zero_division=0))
            logistic_cm_TD_75_category = confusion_matrix(Y_TD_test_category_75, logistic_y_pred_TD_category_75)
            print("Confusion Matrix (Categorization): ")
            print(logistic_cm_TD_75_category)

        # Decision Tree
        if Model_type == 3:
            print("\nFault Categorization Performance Metrics for Decision Tree Model:")
            print(classification_report(Y_TD_test_category_75,dtree_y_pred_TD_category_75, zero_division=0))
            dtree_cm_TD_75_category = confusion_matrix(Y_TD_test_category_75,dtree_y_pred_TD_category_75)
            print("Decision Tree Confusion matrix: ")
            print(dtree_cm_TD_75_category)

        # SVM
        if Model_type == 4:
            print("\nFault Categorization Performance Metrics for SVM Model:")
            print(classification_report(Y_TD_test_category_75,svm_y_pred_TD_category_75,zero_division=0))
            svm_cm_TD_75_category = confusion_matrix(Y_TD_test_category_75,svm_y_pred_TD_category_75)
            print("SVM Confusion matrix: ")
            print(svm_cm_TD_75_category)

        #NN
        if Model_type == 5:
            print("\nFault Categorization Performance Metrics for Neural Network Model:")
            print(classification_report(Y_TD_test_category_75, NN_y_pred_TD_category_75))
            cm_FF_75_category_NN = confusion_matrix(Y_TD_test_category_75,NN_y_pred_TD_category_75)
            print("Confusion matrix: ")
            print(cm_FF_75_category_NN)

    
    answer = -1
    while answer > 1 or answer < 0:
        print("If you would like to look at a different model please type 1 otherwise type 0:\n")
        answer = input()
        answer = int(answer)

        if answer == 1:
            cont = True
        elif answer == 0:
            cont = False
        else:
            print("Error: Incorrect input")
            print("Please type 1 or 0")
