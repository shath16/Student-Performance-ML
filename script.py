import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn import tree 
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow import keras
from IPython.display import display



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
#  Task 1 - Data Preparation
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

print()
print()
print(" ############ Task 1 - Data Preparation ############")

pass_score = 8

#--------------cleaning and preprocessing data set 
#loading student dataset into dataframe
studentsDF = pd.read_csv("Student_Math_Data.csv")

print("\n --* Understanding the Dataset *---")
print(f"Total students: {studentsDF.shape[0]}")
print(f"Total attributes: {studentsDF.shape[1]}")
print(f"\nGrade distributions:")
print(f"G1 (Semester 1): Mean={studentsDF['G1'].mean():.2f}, Std={studentsDF['G1'].std():.2f}")
print(f"G2 (Semester 2): Mean={studentsDF['G2'].mean():.2f}, Std={studentsDF['G2'].std():.2f}")
print(f"G3 (Final): Mean={studentsDF['G3'].mean():.2f}, Std={studentsDF['G3'].std():.2f}")

print("\n --* dataset's shape *---")
print("Number of rows and columns are = ", studentsDF.shape)
 
#display the first 5 rows of dataset
print("\n --* 5 rows of the dataset *---")
print(studentsDF.head()) 


#display summary of dataset 
print("\n --* dataset information *---")
studentsDF.info() 

#histogram to visualise the distribution
plt.figure(figsize=(6, 4))
plt.hist(studentsDF["G3"], bins=10)
plt.axvline(pass_score, linestyle="--", label="Pass threshold (40%)")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Assessment Grades (G3)")
plt.legend()
plt.show()

#--------------removing any missing values 
#checking each column if it contains at least one missing value 
print(" \n --* NaN check: columns *---")
print (studentsDF.isnull(). any())  


#counting number of missing values in each column 
print(" \n --* NaN count per column *---")
print (studentsDF.isnull(). sum())  

#how many NaN values distribute across the rows of the dataframe  
print(" \n --* Number of rows affected by NaN *---")
missing_values = studentsDF.isnull().T.any().sum()
print("Total affected=", missing_values)


#removes rows that do have NaN values 
if missing_values > 0 :
    before_rows = studentsDF.shape[0] 
    studentsDF = studentsDF.dropna()
    after_rows = studentsDF.shape[0] 

    dropped_rows = before_rows - after_rows 

    print(" \n --* Number of rows affected by NaN *---")
    print("Rows before are = ", before_rows)
    print("Row after are = ", after_rows) #calculating how many rows were affected by NaNs  
    print("\n Dropped rows that have NaNs are =", dropped_rows)  
          
else:
    print("\n There are no missing values in any rows")



#converting catergorical to numerical form 
#creating a summary of both nominal and numerical variables 
print(" \n --* summary of all variables (including non-numercal) *---")
print(studentsDF.describe(include='all'))


y = studentsDF["G3"] #this is the target=output variables 
X = studentsDF.drop("G3", axis=1) #removes target variable so only features remain=input variables 
print(" \n --* summary of target only (G3) *---")
print(y.describe())




#identifying nominal/catergorical and numerical columns 
nom_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["number"]).columns

print("\n --* numerical columns names *---")
print(num_cols)

print("\n --* nominal/catergorical columns names *---")
print(nom_cols)


#one-hot encoding

#catergorical columns are only selected 
catergorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

#making a onehotencoder to convert into binary and using it 
encoder = OneHotEncoder (sparse_output= False, handle_unknown= 'ignore')
one_hot_encoded = encoder.fit_transform (X[catergorical_cols])

#the encoded output is converted into an dataframe 
one_hot_DF = pd.DataFrame( one_hot_encoded, columns = encoder.get_feature_names_out(catergorical_cols), index = X.index)

#encoded columns and numerical columns are put together
X_encoded_df = pd.concat([X.drop(catergorical_cols, axis= 1) ,  one_hot_DF] ,  axis =1 )

#gobal split for DT and MLP
train_set, test_set = train_test_split(
    X_encoded_df.index, test_size=0.3, random_state=0
)

#validation split for DT and MLP for training
train_set, val_set = train_test_split(
    train_set, test_size=0.2, random_state=0
)


#showing few rows of encoded output feature set 
print (X_encoded_df.shape)
print(X_encoded_df.head())





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
#  Task 2 - Decision Tree Modelling
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print()
print()
print(" ############ Task 2 - Decision Tree Modelling ############")

task4_rows = []
task2_rows = []


y_tree = y

#for predictions A,B and C 
A_X_tree = X_encoded_df.drop(["G1", "G2"], axis = 1 ) #using up to 30 attributes
B_X_tree = X_encoded_df.drop(["G2"], axis = 1 ) #using up to 31 attributes
C_X_tree = X_encoded_df #using up to 32 attributes


tree_models = {
    "A(no G1 + no G2)": A_X_tree,
    "B(with G1 + no G2)": B_X_tree,
    "C(with G1 + with G2)": C_X_tree
    }


#the best max_depth check
X_tune = C_X_tree
y_tune = y_tree

X_train_tune = C_X_tree.loc[train_set]
y_train_tune = y_tree.loc[train_set]

X_test_tune  = C_X_tree.loc[val_set]
y_test_tune  = y_tree.loc[val_set]

depth_candidates = [2, 3, 4, 5, 6]
dt_depth_rows = []

for d in depth_candidates:
    dt_tmp = tree.DecisionTreeRegressor(max_depth=d, random_state=0)
    dt_tmp.fit(X_train_tune, y_train_tune)
    y_tmp = dt_tmp.predict(X_test_tune)

    rmse_tmp = np.sqrt(metrics.mean_squared_error(y_test_tune, y_tmp))

    actual_pass_tmp = (y_test_tune >= pass_score).astype(int)
    pred_pass_tmp = (y_tmp >= pass_score).astype(int)
    acc_tmp = (actual_pass_tmp == pred_pass_tmp).mean()

    dt_depth_rows.append({
        "max_depth": d,
        "RMSE": rmse_tmp,
        "Pass/Fail Accuracy": acc_tmp
    })

dt_depth_df = pd.DataFrame(dt_depth_rows).round(3)

print("\nDecision Tree depth experiment (Scenario C):")
display(dt_depth_df.sort_values("max_depth").reset_index(drop=True))

#plotting (RMSE vs depth)
plt.figure(figsize=(6, 4))
plt.plot(dt_depth_df["max_depth"], dt_depth_df["RMSE"], marker="o")
plt.xlabel("max_depth")
plt.ylabel("RMSE")
plt.title("Decision Tree Depth vs RMSE (Scenario C)")
plt.grid(True)
plt.show()

#showing the best depth
best_row = dt_depth_df.loc[dt_depth_df["RMSE"].idxmin()]

best_depth = int(best_row["max_depth"])
print(f"Final DT models below are trained with max_depth = {best_depth}.")





#LOOP
for model_name, X_tree in tree_models.items():

    print("\n --* DECISION TREE MODEL = *--- ", model_name)

    #splitting dataset into training set and test set 
    X_train = X_tree.loc[train_set]
    X_test  = X_tree.loc[test_set]
    y_train = y_tree.loc[train_set]
    y_test  = y_tree.loc[test_set]
    #split into 70% training and 30% test


    #training a decision tree regressor 
    dt_model = tree.DecisionTreeRegressor(max_depth=best_depth, random_state=0)
    tree_g3 = dt_model.fit( X_train, y_train )

    #predicting g3 on test set 
    y_pred = tree_g3.predict(X_test)

    #evaluating prediciton accuracy with regression metrics 
    testMAE = metrics.mean_absolute_error( y_test, y_pred )
    testMSE = metrics.mean_squared_error(y_test, y_pred)
    testRMSE = np.sqrt(testMSE)


    print(" \n - results for decision tree regressor - ")
    print("mean absolute error (MAE) =", testMAE)
    print("mean squared error (MSE) =" , testMSE)
    print("root mean squared error (RMSE)=", testRMSE)


    #converting the predicted scores into pass/fail
    actual_pass = (y_test >= pass_score).astype(int)
    predicted_pass = ( y_pred >= pass_score ).astype(int)

    #permutation feature importance 
    perm_result = permutation_importance(
        tree_g3,
        X_test,
        y_test,
        n_repeats=10,
        random_state=0,
        scoring="neg_mean_squared_error"
    )


    importances = pd.Series(
        perm_result.importances_mean,
        index=X_test.columns
    ).sort_values(ascending=False)

    print("\nTop 10 important features (permutation importance):")
    print(importances.head(10))

    importances.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Top 10 DT Feature Importances – {model_name}")
    plt.xlabel("Mean decrease in model performance")
    plt.show()
    
    #decision tree displayed 
    fig = plt.figure(figsize = (25,12))
    tree.plot_tree(tree_g3, feature_names=X_train.columns, max_depth=3)

    plt.title(f"decision tree model {model_name}")
    plt.show()

    print("pass mark = ", pass_score )
    print("actual pass percentage =", actual_pass.mean() )
    print("predicted pass percentage =",predicted_pass.mean() )

    #confusion matrix for pass or fail 
    cm = metrics.confusion_matrix(actual_pass.astype(int), predicted_pass.astype(int))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
    plt.figure(figsize = (6,4))
    cm_display.plot()
    plt.title(f"DT Pass/Fail Confusion Matrix-{model_name}")
    plt.show()


    # --- Task 4 colllecting rows (DT) ---
    pf_accuracy = (actual_pass == predicted_pass).mean()
    
    row = {
        "Approach": "DT",
        "Scenario": model_name,
        "RMSE": testRMSE,
        "Pass/Fail Accuracy": pf_accuracy,
        "Actual Pass %": actual_pass.mean(),
        "Predicted Pass %": predicted_pass.mean(),
        "Pass % Error": abs(actual_pass.mean() - predicted_pass.mean())
    }
    
    task2_rows.append({k: row[k] for k in row if k != "Approach"})
    task4_rows.append(row) 



#display
print()
print("*----summary display of result----*")
task2_df = (pd.DataFrame(task2_rows).round(3).sort_values("Scenario").reset_index(drop=True))
display(task2_df)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# Task 3 - MLP Modelling
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


print()
print()
print(" ############ Task 3 - MLP Modelling ############")

task3_rows = []

y_mlp = y  # target

#for predictions A,B and C 
A_X_mlp = X_encoded_df.drop(["G1", "G2"], axis=1)  #using up to 30 attributes
B_X_mlp = X_encoded_df.drop(["G2"], axis=1)        #using up to 31 attributes
C_X_mlp = X_encoded_df                             #using up to 32 attributes

mlp_models = {
    "A(no G1 + no G2)": A_X_mlp,
    "B(with G1 + no G2)": B_X_mlp,
    "C(with G1 + with G2)": C_X_mlp
}

#LOOP
for model_name, X_mlp in mlp_models.items():
    print(" \n --* MLP MODEL = *---", model_name)

    X_train = X_mlp.loc[train_set]
    X_val   = X_mlp.loc[val_set]
    X_test  = X_mlp.loc[test_set]

    y_train = y_mlp.loc[train_set]
    y_val   = y_mlp.loc[val_set]
    y_test  = y_mlp.loc[test_set]


    #normalisation using MinMaxScaler 
    mlpInputsScaler = MinMaxScaler()
    mlpFeat_trainScaled = mlpInputsScaler.fit_transform(X_train)
    mlpFeat_valScaled   = mlpInputsScaler.transform(X_val)
    mlpFeat_testScaled  = mlpInputsScaler.transform(X_test)

    mlp_model = keras.models.Sequential([
        keras.layers.Dense(
            20, activation="relu",
            input_shape=mlpFeat_trainScaled.shape[1:]
        ),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])

    #compliling 
    mlp_model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.SGD(learning_rate=0.01)
    )
    
    #early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    tf.keras.backend.clear_session()

    #training validationset - used for early stopping
    executionhistory = mlp_model.fit(
        mlpFeat_trainScaled,
        y_train.to_numpy(),
        epochs=100,
        validation_data=(mlpFeat_valScaled, y_val.to_numpy()),
        callbacks=[early_stop],
        verbose=1
    )

    print()
    print()
    plt.figure(figsize=(6,4))
    plt.plot(executionhistory.history["loss"], label="train loss")
    plt.plot(executionhistory.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"MLP Training Loss - {model_name}")
    plt.legend()
    plt.show()

    #predicting on test set only
    y_MLP_pred = mlp_model.predict(mlpFeat_testScaled, verbose=0).reshape(-1)

    #regression metrics
    testMAE = metrics.mean_absolute_error(y_test, y_MLP_pred)
    testMSE = metrics.mean_squared_error(y_test, y_MLP_pred)
    testRMSE = np.sqrt(testMSE)

    print()
    print()
    print(" \n - results for MLP regressor -")
    print("mean absolute error (MAE) =", testMAE)
    print("mean squared error (MSE) =" , testMSE)
    print("root mean squared error (RMSE)=", testRMSE)

    #MLP pass/fail confusion matrix 
    actual_passMLP = (y_test >= pass_score).astype(int)
    predicted_passMLP = (y_MLP_pred >= pass_score).astype(int)

    cm_mlp = metrics.confusion_matrix(actual_passMLP, predicted_passMLP)
    cm_display_mlp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=["Fail", "Pass"])
    cm_display_mlp.plot()
    plt.title(f"MLP Pass/Fail Confusion Matrix-{model_name}")
    plt.show()

    print("pass mark = ", pass_score)
    print("actual pass percentage =", actual_passMLP.mean())
    print("predicted pass percentage =", predicted_passMLP.mean())

    # --- Task 4 collecting rows (MLP) ---
    pf_accuracy = (actual_passMLP == predicted_passMLP).mean()
    
    row = {
        "Approach": "MLP",
        "Scenario": model_name,
        "RMSE": testRMSE,
        "Pass/Fail Accuracy": pf_accuracy,
        "Actual Pass %": actual_passMLP.mean(),
        "Predicted Pass %": predicted_passMLP.mean(),
        "Pass % Error": abs(actual_passMLP.mean() - predicted_passMLP.mean()),
    }
    
    task3_rows.append({k: row[k] for k in row if k != "Approach"})
    task4_rows.append(row)

#display
print()
print("*----summary display of result----*")
task3_df = (pd.DataFrame(task3_rows).round(3).sort_values("Scenario").reset_index(drop=True))
display(task3_df)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# Task 4 - Model Evaluation
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print()
print()
print(" ############ Task 4 - Model Evaulation ############")

# Task 4 - Model Evaluation - Summary Table
task4_df = (
    pd.DataFrame(task4_rows)
    .drop_duplicates(subset=["Approach", "Scenario"], keep="last")
    .round(3)
    .sort_values(["Scenario", "Approach"])
    .reset_index(drop=True)
)

display(task4_df)
