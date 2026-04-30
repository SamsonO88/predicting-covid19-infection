# import libraries algorithm training
from sklearn.model_selection import cross_validate, train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.utils import resample
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv(r"C:\Users\Dell\Documents\my_linux\repos_\predicting-covid19-infection\predicting_covid19_infection\data\raw\cleaned_covid.csv")
SEED = 42

# check for duplicates
df = df[df["Result"] != "INDETERMINATE"]
print(df.duplicated().sum())

df.head(2)

df.columns

"""#### Feature engineering and transformation"""

def label_encode_columns(df, columns):
    """
    Label encodes the specified categorical columns in a DataFrame without using sklearn.

    Args:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to label encode.

    Returns:
    pd.DataFrame: DataFrame with label-encoded columns.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes  # Convert to category codes
    return df

# Columns to encode
columns_to_encode = ['Chest pain', 'Cough', 'Diarrhea','Fatigue or general weakness', 'Fever', 'Headache',
                    'Thorax (sore throat)', 'Nausea', 'Runny nose', 'Sore throat or pharyngitis', 'Vomiting',
                    'Loss of Taste', 'Loss of Smell', 'Result']

# Apply function
df_encoded = label_encode_columns(df, columns_to_encode)

print(df_encoded)

def one_hot_encode_columns(df, columns):
    """
    One-hot encodes the specified categorical columns in a DataFrame without using sklearn.

    Args:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to one-hot encode.

    Returns:
    pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    return pd.get_dummies(df, columns=columns, drop_first=False)  # Keep all categories

# Columns to encode
columns_to_encode = ['Sex']

# Apply function
df_encoded = one_hot_encode_columns(df_encoded, columns_to_encode)

df_encoded.head(2)

# balancing the imbalance dataset
# Print class distribution
print("Original Class Distribution:\n", df_encoded['Result'].value_counts())

# Separate majority and minority classes
classes = df_encoded['Result'].value_counts().index  # Get unique classes
majority_class = df_encoded['Result'].value_counts().idxmax()  # Find majority class
max_size = df_encoded['Result'].value_counts().max()  # Size of majority class

# Upsample minority classes
df_balanced = pd.concat([
    resample(df_encoded[df_encoded['Result'] == cls], replace=True, n_samples=max_size, random_state=42)
    if cls != majority_class else df_encoded[df_encoded['Result'] == cls]
    for cls in classes
])

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Print new class distribution
print("\nBalanced Class Distribution:\n", df_balanced['Result'].value_counts())

#
X = df_balanced.drop(["Result", "Birth Year"], axis = 1)
y = df_balanced["Result"]
y.value_counts()

X.columns

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = SEED)

# Scale the Data (Normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""#### Experimentation without cross validation"""

# initialize the algorithms for training
lrc = LogisticRegression()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()

"""##### Logistics regression"""

# train a Logistic Regression Model
lrc.fit(X_train, y_train)

# Make Predictions
y_pred = lrc.predict(X_test_scaled)

class_report = classification_report(y_test, y_pred)
# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("\n Accuracy of Logistics Regression:\n", accuracy_score(y_test, y_pred))
print("\n Precision of Logistics Regression:\n", precision_score(y_test, y_pred))
print("\n Recall of Logistics Regression:\n", recall_score(y_test, y_pred))
print("\n F1_score of Logistics Regression:\n", f1_score(y_test, y_pred))
print("\n roc_au of Logistics Regression:\n", roc_auc)
print("\n Logistics Regression Classification Report:\n", class_report)

"""##### decision tree classifier"""

# train a Decision Tree Model
dtc.fit(X_train, y_train)

# Make Predictions
y_pred_dtc = dtc.predict(X_test_scaled)

# ROC Curve and AUC
fpr_dtc, tpr_dtc, thresholds_dtc = roc_curve(y_test, y_pred_dtc)
roc_auc_dtc = auc(fpr_dtc, tpr_dtc)
# classification report
class_report_dtc = classification_report(y_test, y_pred_dtc)
print("\n Accuracy of Decision Tree:\n", accuracy_score(y_test, y_pred_dtc))
print("\n Precision of Decision Tree:\n", precision_score(y_test, y_pred_dtc))
print("\n Recall of Decision Tree:\n", recall_score(y_test, y_pred_dtc))
print("\n F1_score of Decision Tree:\n", f1_score(y_test, y_pred_dtc))
print("\n roc_auc of Decision Tree:\n", roc_auc_dtc)
print("\nDecision Tree Classification Report:\n", class_report_dtc)

# Export the trained model
joblib.dump(dtc, r"C:\Users\Dell\Documents\my_linux\repos_\predicting-covid19-infection\predicting_covid19_infection\models\decision_tree_model.pkl")

print("Model saved successfully as 'decision_tree_model.pkl'")

"""##### random forest classifier"""

# train a Random Forest Model
rfc.fit(X_train, y_train)

# Make Predictions
y_pred_rfc = rfc.predict(X_test_scaled)

# ROC Curve and AUC
fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_test, y_pred_rfc)
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
# classification report
class_report_rfc = classification_report(y_test, y_pred_rfc)
print("\n Accuracy of Random Forest:\n", accuracy_score(y_test, y_pred_rfc))
print("\n Precision of Random Forest:\n", precision_score(y_test, y_pred_rfc))
print("\n Recall of Random Forest:\n", recall_score(y_test, y_pred_rfc))
print("\n F1_score of Random Forest:\n", f1_score(y_test, y_pred_rfc))
print("\n roc_auc of Random Forest:\n", roc_auc_rfc)
print("\n Random Forest Classification Report:\n", class_report_rfc)

# Export the trained model
joblib.dump(rfc, r"C:\Users\Dell\Documents\my_linux\repos_\predicting-covid19-infection\predicting_covid19_infection\models\random_forest_model.pkl")

print("Model saved successfully as 'random_forest_model.pkl'")

"""#### Experimentation with cross validation"""

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state = SEED)

# Scale Features
X_scaled = scaler.fit_transform(X)

"""##### cross validation with logisitic regression"""

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold split
scores = cross_val_score(lrc, X_scaled, y, cv=kf, scoring='f1_macro')  # Compute f1-score for each fold

# Print Results
print(f"Cross-Validation Scores: {scores}")
print(f"Mean f1 score: {np.mean(scores):.2f}")

"""##### cross validation with decision tree"""

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold split
scores_dtc = cross_val_score(dtc, X_scaled, y, cv=kf, scoring='f1_macro')  # Compute f1-score for each fold

# Print Results
print(f"Cross-Validation Scores: {scores_dtc}")
print(f"Mean f1 score: {np.mean(scores_dtc):.2f}")

"""##### cross validation with random forest classifier"""

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold split
scores_rfc = cross_val_score(rfc, X_scaled, y, cv=kf, scoring='f1_macro')  # Compute f1-score for each fold

# Print Results
print(f"Cross-Validation Scores: {scores_rfc}")
print(f"Mean f1 score: {np.mean(scores_rfc):.2f}")

X.columns

# Train the model on the full dataset
rfc.fit(X_scaled, y)

# Export the trained model
joblib.dump(rfc, r"C:\Users\Dell\Documents\my_linux\repos_\predicting-covid19-infection\predicting_covid19_infection\models\random_forest_model.pkl")

print("Model saved successfully as 'random_forest_model.pkl'")