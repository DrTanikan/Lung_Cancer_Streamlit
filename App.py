import numpy as np
import pandas as pd
import streamlit as st
import pickle as pk

# Load model
model = pk.load(open("LungCancer.pkl", "rb"))

st.header("LungForesight: AI-Based Lung Cancer Risk Prediction")

# Inputs (ตรงกับ dataset)
Gender = st.number_input("Gender (0 = Female, 1 = Male)", 0, 1, 1)
Age = st.number_input("Age", min_value=0, step=1)
Smoke = st.number_input("Smoke (0 = No, 1 = Yes)", 0, 1, 1)
Yellow_Finger = st.number_input("Yellow Finger (0 = No, 1 = Yes)", 0, 1, 1)
Anxiety = st.number_input("Anxiety (0 = No, 1 = Yes)", 0, 1, 1)
Peer_Pressure = st.number_input("Peer Pressure (0 = No, 1 = Yes)", 0, 1, 1)
Chronic_Disease = st.number_input("Chronic Disease (0 = No, 1 = Yes)", 0, 1, 1)
Fatigue = st.number_input("Fatigue (0 = No, 1 = Yes)", 0, 1, 1)
Allergy = st.number_input("Allergy (0 = No, 1 = Yes)", 0, 1, 1)
Wheezing = st.number_input("Wheezing (0 = No, 1 = Yes)", 0, 1, 1)
Alcohol = st.number_input("Alcohol (0 = No, 1 = Yes)", 0, 1, 1)
Coughing = st.number_input("Coughing (0 = No, 1 = Yes)", 0, 1, 1)
Shortness_of_Breath = st.number_input("Shortness Of Breath (0 = No, 1 = Yes)", 0, 1, 1)
Swallowing_Difficulty = st.number_input("Swallowing Difficulty (0 = No, 1 = Yes)", 0, 1, 1)
Chest_Pain = st.number_input("Chest Pain (0 = No, 1 = Yes)", 0, 1, 1)

# Predict
if st.button("Predict"):
    X_input = np.array([[
        Gender,
        Age,
        Smoke,
        Yellow_Finger,
        Anxiety,
        Peer_Pressure,
        Chronic_Disease,
        Fatigue,
        Allergy,
        Wheezing,
        Alcohol,
        Coughing,
        Shortness_of_Breath,
        Swallowing_Difficulty,
        Chest_Pain
    ]])

    # Prediction
    output = model.predict(X_input)

    # Probability (ถ้า model รองรับ)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)
        risk_percent = proba[0][1] * 100

        st.write(f"Lung cancer probability: {risk_percent:.2f}%")

        if output[0] == 1:
            st.error(f"⚠️ High risk of lung cancer ({risk_percent:.2f}%)")
        else:
            st.success(f"✅ Low risk of lung cancer ({risk_percent:.2f}%)")