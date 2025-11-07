import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IDS Demo", layout="wide")
st.title("🚨 نظام كشف التسلل (IDS) - عرض توضيحي")
st.write("ارفع ملف CSV يحتوي على بيانات الشبكة ليتم التنبؤ إذا كانت حركة طبيعية أو هجوم.")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# File uploader
uploaded_file = st.file_uploader("اختر ملف CSV", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("معاينة البيانات")
        st.dataframe(df.head())

        # Scale features
        X_scaled = scaler.transform(df.values)

        # Predict
        preds = model.predict(X_scaled)
        df['Prediction'] = ["Normal" if p=="normal" else "Attack" for p in preds]

        st.subheader("النتائج")
        st.dataframe(df)

        st.subheader("ملخص")
        st.write(df['Prediction'].value_counts())

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="تحميل النتائج (CSV)",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"حدث خطأ: {e}\nتأكد أن ملف CSV يحتوي على نفس الأعمدة المستخدمة أثناء التدريب.")
