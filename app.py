import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# Expected columns
expected_columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]

st.title("Intrusion Detection System (IDS) - NSL-KDD Cloud Demo")
st.write("Upload NSL-KDD formatted CSV file (without label & level columns) to detect attacks")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Check columns match
    if list(data.columns) != expected_columns:
        st.error("❌ Column mismatch! Please upload data with the correct NSL-KDD feature columns only (without label & level).")
        st.write("✅ Expected columns:")
        st.code("\n".join(expected_columns))
    else:
        st.success("✅ File format correct")

        st.write("### Sample Data")
        st.write(data.head())

        # Predict
        preds = model.predict(data)

        # Add result column
        result = pd.DataFrame(preds, columns=["Prediction"])
        result["Prediction"] = result["Prediction"].map({0:"Normal", 1:"Attack"})

        st.write("### Prediction Results")
        st.write(result.head())

        # Download results
        output = result.to_csv(index=False).encode("utf-8")
        st.download_button("Download Prediction Results", output, "IDS_Output.csv", "text/csv")
