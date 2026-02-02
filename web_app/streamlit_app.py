# ==============================================================================
# --- Import libraries ---
# ==============================================================================

import streamlit as st
import requests



# ==============================================================================
# --- Config ---
# ==============================================================================

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Fraud Guard 2026",
    page_icon="🛡️",
    layout="centered"
)



# ==============================================================================
# --- Header ---
# ==============================================================================

st.title("🛡️ Fraud Guard 2026")
st.caption("Real-time transaction risk assessment powered by ML")

st.divider()



# ==============================================================================
# --- Transaction Input ---
# ==============================================================================

with st.form("transaction_form"):

    st.subheader("Transaction Details")

    user_id = st.text_input("User ID", value="USER_123")

    amount = st.number_input(
        "Transaction Amount",
        min_value=1.0,
        value=120.0,
        step=1.0
    )

    col1, col2 = st.columns(2)

    with col1:
        lat = st.slider(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=34.05
        )

    with col2:
        lon = st.slider(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-118.24
        )

    auth_method = st.selectbox(
        "Authentication Method",
        ["Biometric", "PIN", "Password"]
    )

    category = st.selectbox(
        "Transaction Category",
        ["food", "grocery", "tech", "travel", "utilities", "entertainment"]
    )

    time_delta_min = st.number_input(
        "Minutes Since Last Transaction",
        min_value=1.0,
        value=10.0
    )

    tx_count_24h = st.number_input(
        "How many transactions in the last 24 hours?",
        min_value=0,
        value=3
    )

    submitted = st.form_submit_button("🔍 Analyze Transaction")



# ==============================================================================
# --- Prediction ---
# ==============================================================================

if submitted:

    payload = {
        "user_id": user_id,
        "amount": amount,
        "lat": lat,
        "lon": lon,
        "auth_method": auth_method,
        "category": category,
        "time_delta_min": time_delta_min,
        "tx_count_24h": tx_count_24h
    }

    with st.spinner("Running fraud checks..."):

        try:
            response = requests.post(API_URL, json=payload, timeout=5)

            if response.status_code != 200:
                st.error(f"API Error: {response.text}")
            else:
                result = response.json()

                st.divider()
                st.subheader("🧠 Model Decision")

                prob = result["fraud_probability"]
                prediction = result["prediction"]
                model_version = result["model_version"]

                # Risk visualization
                st.metric(
                    label="Fraud Probability",
                    value=f"{prob * 100:.2f} %"
                )

                if prediction == "fraud":
                    st.error("🚨 HIGH RISK TRANSACTION")
                else:
                    st.success("✅ Transaction appears legitimate")

                st.caption(f"Model Version: `{model_version}`")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

