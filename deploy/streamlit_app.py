import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv


load_dotenv()

st.title("Telecom Customer Churn Prediction")

API_URL = os.getenv("API_URL","http://localhost:8000/churn-predict")

st.header("Customer Details")

# Minimal inputs (you can expand later)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
plan = st.selectbox("Plan", ["Basic", "Express", "Premium"])
nps = st.number_input("NPS Score", min_value=0, max_value=10, value=5)
page_views = st.number_input("Page Views", min_value=0, value=20)
logins = st.number_input("Logins", min_value=0, value=5)
rating = st.number_input("Rating", min_value=1, max_value=5, value=3)

if st.button("Predict Churn"):
    payload = {
        "records": [
            {
                "Age": age,
                "Gender": gender,
                "Plan": plan,
                "NPS": nps,
                "PageViews": page_views,
                "Logins": logins,
                "Rating": rating
            }
        ]
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction Result")

            st.write(f"**Prediction:** {result['prediction_class']}")
            st.write(f"**Churn Probability:** {result['churn_probability']:.2f}")
            st.write(f"**Non-Churn Probability:** {result['non_churn_probability']:.2f}")

            if result["prediction_label"] == 1:
                st.error("⚠️ Customer is likely to churn")
            else:
                st.success("✅ Customer is likely to stay")
        else:
            st.error(f"API error: {response.status_code}")

    except Exception as e:
        st.error(str(e))