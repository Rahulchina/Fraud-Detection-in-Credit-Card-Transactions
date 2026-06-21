import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(
    r"C:\Users\rahul\Downloads\archive (6)\creditcard.csv"
)

print(df.head())
print(df.info())
print(df["Class"].value_counts())

# Missing Values
print(df.isnull().sum())

# Scale Amount
scaler = StandardScaler()

df["Amount"] = scaler.fit_transform(
    df[["Amount"]]
)

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Balance Dataset
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

normal_sample = normal.sample(
    n=len(fraud),
    random_state=42
)

balanced_df = pd.concat(
    [fraud, normal_sample]
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
)

X_balanced = balanced_df.drop(
    "Class",
    axis=1
)

y_balanced = balanced_df["Class"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.2,
    random_state=42
)

# Isolation Forest
iso = IsolationForest(
    contamination=0.01,
    random_state=42
)

iso.fit(X_train)

pred_iso = iso.predict(X_test)

pred_iso = np.where(
    pred_iso == -1,
    1,
    0
)

print("\nIsolation Forest")
print(confusion_matrix(y_test, pred_iso))

# Local Outlier Factor
lof = LocalOutlierFactor(
    contamination=0.01,
    novelty=True
)

lof.fit(X_train)

pred_lof = lof.predict(X_test)

pred_lof = np.where(
    pred_lof == -1,
    1,
    0
)

print("\nLocal Outlier Factor")
print(confusion_matrix(y_test, pred_lof))

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(
    X_train,
    y_train
)

# Save Model
with open("xgb_model.pkl", "wb") as file:
    pickle.dump(xgb, file)

print("\nModel Saved Successfully")

# Predictions
y_pred = xgb.predict(X_test)

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred
))

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
y_prob = xgb.predict_proba(
    X_test
)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


import streamlit as st
import pandas as pd
import pickle

# Load Model
model = pickle.load(
    open("xgb_model.pkl", "rb")
)

st.title("Credit Card Fraud Detection System")

st.write(
    "Enter transaction details below"
)

# Input Features
time = st.number_input("Time")

features = [time]

for i in range(1, 29):
    value = st.number_input(
        f"V{i}",
        value=0.0
    )
    features.append(value)

amount = st.number_input(
    "Amount",
    value=0.0
)

features.append(amount)

if st.button("Predict Fraud"):

    columns = (
        ["Time"] +
        [f"V{i}" for i in range(1,29)] +
        ["Amount"]
    )

    input_df = pd.DataFrame(
        [features],
        columns=columns
    )

    prediction = model.predict(
        input_df
    )

    probability = model.predict_proba(
        input_df
    )[0][1]

    if prediction[0] == 1:
        st.error(
            f"Fraud Detected\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"Legitimate Transaction\n\nProbability: {(1-probability):.2%}"
        )
