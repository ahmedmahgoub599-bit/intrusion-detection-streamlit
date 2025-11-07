import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# تحميل النموذج والمقياس
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚨 Intrusion Detection System (IDS)")
st.write("Upload network traffic CSV to detect **Normal vs Attack**")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # -----------------------------
    # قراءة البيانات
    # -----------------------------
    df = pd.read_csv(uploaded_file)

    st.write("📄 **Preview uploaded data:**")
    st.dataframe(df.head())

    # -----------------------------
    # إنشاء عمود attack_type إذا غير موجود
    # -----------------------------
    if "label" in df.columns:
        df["attack_type"] = df["label"].apply(lambda x: "normal" if x=="normal" else "attack")
    elif "attack_type" not in df.columns:
        df["attack_type"] = "unknown"

    # -----------------------------
    # تحويل categorical مثل التدريب
    # -----------------------------
    cat_cols = ["protocol_type", "service", "flag"]
    for col in cat_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # -----------------------------
    # حذف أعمدة غير مهمة
    # -----------------------------
    for col in ["label", "level"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # -----------------------------
    # تجهيز أعمدة النموذج المتوقعة
    # -----------------------------
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = df.columns.tolist()  # fallback

    # -----------------------------
    # إضافة الأعمدة الناقصة بـ 0
    # -----------------------------
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # -----------------------------
    # حذف الأعمدة الزائدة
    # -----------------------------
    df = df[expected_cols]

    # -----------------------------
    # تحويل القيم إلى float لتجنب مشاكل StandardScaler
    # -----------------------------
    df = df.apply(pd.to_numeric, errors='ignore')

    # -----------------------------
    # تطبيق StandardScaler (تحويل إلى numpy لتجاوز فحص feature_names)
    # -----------------------------
    X_scaled = scaler.transform(df.to_numpy())

    # -----------------------------
    # التنبؤ
    # -----------------------------
    predictions = model.predict(X_scaled)
    df["Prediction"] = predictions

    st.write("✅ **Prediction Results:**")
    st.dataframe(df[["Prediction"]].head())

    # -----------------------------
    # ملخص النتائج
    # -----------------------------
    st.write("📊 **Summary:**")
    st.write(df["Prediction"].value_counts())

else:
    st.info("⬆️ Please upload a CSV file to start analysis.")
