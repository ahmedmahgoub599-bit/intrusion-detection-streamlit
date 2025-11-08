import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.title("Intrusion Detection System - IDS")

uploaded_file = st.file_uploader("Upload CSV file for prediction")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    predictions = model.predict(data)
    st.write(predictions)
