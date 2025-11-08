import pandas as pd
import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("Intrusion Detection System (IDS) - Cloud Demo")
st.write("Upload CSV file (with raw values for protocol_type, service, flag)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Columns to encode
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop label/level if present
    df_encoded = df_encoded.drop(columns=[col for col in ['label','level'] if col in df_encoded.columns])
    
    # Ensure same columns as training (add missing columns with 0)
    training_columns = joblib.load("model_columns.pkl")  # يجب أن تحفظ قائمة الأعمدة المستخدمة في التدريب
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[training_columns]
    
    st.write("### Sample Encoded Data")
    st.write(df_encoded.head())
    
    # Predict
    preds = model.predict(df_encoded)
    df['Prediction'] = preds
    df['Prediction'] = df['Prediction'].map({0: "Normal", 1: "Attack"})
    
    st.write("### Prediction Results")
    st.write(df[['Prediction']].head())
    
    # Download predictions
    output = df[['Prediction']].to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", output, "IDS_Output.csv", "text/csv")
