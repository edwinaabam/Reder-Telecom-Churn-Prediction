import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys

# 1. FIX THE FOLDER PATH: This tells Python to look in the root folder for DataCleaning
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from DataCleaning.clean import clean_data
except ModuleNotFoundError:
    st.error("Could not find the DataCleaning module. Ensure the folder is in your GitHub root.")

# --- NAVIGATION SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/antenna.png") # Optional antenna icon
    st.title("Navigation")
    page = st.radio("Go to:", ["Churn Predictor", "About the Model", "Customer Insights"])
    st.info("Reder Telecom v1.0")

# --- COLORED BANNER ---
# This creates a thin, colored bar at the top
st.markdown("""
    <div style="background-color:#FF4B4B; padding:10px; border-radius:5px; margin-bottom:20px;">
        <h3 style="color:white; text-align:center; margin:0;">Reder Telecom Customer Churn Prediction</h3>
    </div>
    """, unsafe_allow_html=True)

# 2. LOAD ASSETS
# Adjusting paths to go up one level from 'deploy/' to find 'model/'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'schema.json')

@st.cache_resource
def load_all():
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return model, schema["model_schema"]

try:
    model, feature_schema = load_all()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

# --- UI INPUTS (The part you keep) ---
st.header("Enter Customer Information")
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

# --- PREDICTION LOGIC ---
if st.button("Run Prediction"):
    # Create the raw dataframe with your FastAPI-style keys
    raw_data = pd.DataFrame([{
        "Age": age, "Gender": gender, "Plan": plan, "NPS": nps,
        "PageViews": page_views, "Logins": logins, "Rating": rating,
        "Frequency": "weekly", "Recency_days": 30, # Default values for cleaning
        "ActionsLast30Days": 5, "num_calls": 1
    }])

    try:
        with st.spinner("Processing through DataCleaning pipeline..."):
            # Clean and reindex to match what the model saw at fit time
            cleaned_data = clean_data(raw_data)
            final_input = cleaned_data.reindex(columns=feature_schema, fill_value=0)

            prediction = model.predict(final_input)[0]
            proba = model.predict_proba(final_input)[0][1]

            st.markdown("---")
            st.metric("Risk of Churn", f"{proba:.2%}")

            if prediction == 1:
                st.error("⚠️ HIGH RISK: This customer is likely to churn.")
            else:
                st.success("✅ LOW RISK: This customer is likely to stay.")
    except Exception as e:
        st.error(f"Pipeline Error: {e}")