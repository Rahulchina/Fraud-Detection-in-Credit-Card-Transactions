import streamlit as st
import pandas as pd
import pickle

model = pickle.load(
    with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)
)

st.title("Credit Card Fraud Detection System")

time = st.number_input("Time")
features = [time]

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    features.append(value)

amount = st.number_input("Amount", value=0.0)
features.append(amount)

if st.button("Predict Fraud"):

    columns = (
        ["Time"] +
        [f"V{i}" for i in range(1, 29)] +
        ["Amount"]
    )

    input_df = pd.DataFrame(
        [features],
        columns=columns
    )

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(
            f"Fraud Detected\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"Legitimate Transaction\nProbability: {(1-probability):.2%}"
        )
