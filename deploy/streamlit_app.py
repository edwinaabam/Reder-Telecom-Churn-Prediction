import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import sys

# Path fix to ensure we can see DataCleaning folder in Docker root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

st.set_page_config(page_title="Reder Telecom Churn", page_icon="📡")
st.title("Telecom Customer Churn Prediction")

# For local testing, it uses .env. For Docker/Spaces, it defaults to localhost:8000
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/churn-predict")

st.header("Customer Details")

# Layout the inputs nicely
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    plan = st.selectbox("Plan", ["Basic", "Express", "Premium"])
    nps = st.number_input("NPS Score", min_value=0, max_value=10, value=5)

with col2:
    page_views = st.number_input("Page Views", min_value=0, value=20)
    logins = st.number_input("Logins", min_value=0, value=5)
    rating = st.number_input("Rating", min_value=1, max_value=5, value=3)

if st.button("Predict Churn"):
    # The dictionary keys here MUST match your model's expected feature names
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
        with st.spinner("Analyzing customer data..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.markdown("---")
            st.subheader("Prediction Result")

            # Display probabilities in a nice way
            st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            
            if result["prediction_label"] == 1:
                st.error(f"⚠️ Result: {result['prediction_class']}")
                st.warning("Recommendation: Proactive retention call recommended.")
            else:
                st.success(f"✅ Result: {result['prediction_class']}")
                st.info("Customer is stable.")
                
        else:
            st.error(f"API error: {response.status_code}. Make sure the backend is running.")

    except Exception as e:
        st.error(f"Connection Error: Is the FastAPI server running at {API_URL}?")