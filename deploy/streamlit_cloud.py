import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys
from PIL import Image

# 1. PATH FIX: Ensures DataCleaning is found from the 'deploy' subfolder
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

try:
    from DataCleaning.clean import clean_data
except ModuleNotFoundError:
    st.error("Module 'DataCleaning' not found. Ensure it is in your GitHub root.")

# --- NAVIGATION SIDEBAR ---
with st.sidebar:
    # --- LOGO SECTION ---
    # Load and display your logo from the deploy folder
    try:
        logo_path = os.path.join(current_dir, "radiologo.png") # Adjust extension if it's .jpg or .svg
        logo = Image.open(logo_path)
        st.image(logo, use_container_width=True)
    except Exception:
        st.warning("Logo file not found in 'deploy' folder.")

    st.title("📡 Reder Menu")
    
    # Dropdown select menu
    navigation = st.selectbox(
        "Select Page",
        ["Churn Predictor", "Model Metrics", "Settings"]
    )
    
    # Expandable option list
    with st.expander("ℹ️ App Information"):
        st.write("This app uses a Random Forest model to analyze customer behavior and predict the likelihood of churn.")
        st.write("**Version:** 1.0.3")

# --- COLORED BANNER (NAVY BLUE) ---
st.markdown("""
    <div style="background-color:#002b5b; padding:15px; border-radius:10px; margin-bottom:25px;">
        <h2 style="color:white; text-align:center; margin:0; font-family:sans-serif;">Reder Telecom Customer Churn Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

# 2. LOAD ASSETS (Model and Schema)
MODEL_PATH = os.path.join(current_dir, '..', 'model', 'model.pkl')
SCHEMA_PATH = os.path.join(current_dir, '..', 'model', 'schema.json')

@st.cache_resource
def load_all():
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return model, schema["model_schema"]

try:
    model, feature_schema = load_all()
except Exception as e:
    st.error(f"Error loading model/schema: {e}")

# --- UI INPUTS ---
st.subheader("Customer Data Entry")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    plan = st.selectbox("Plan", ["Basic", "Express", "Premium"])
    nps = st.slider("NPS Score", 0, 10, 5)

with col2:
    page_views = st.number_input("Page Views", 0, 1000, 20)
    logins = st.number_input("Logins", 0, 100, 5)
    rating = st.selectbox("Rating", [1, 2, 3, 4, 5], index=2)

# --- PREDICTION LOGIC ---
if st.button("Calculate Churn Risk"):
    raw_data = pd.DataFrame([{
        "Age": age, "Gender": gender, "Plan": plan, "NPS": nps,
        "PageViews": page_views, "Logins": logins, "Rating": rating,
        "Frequency": "weekly", "Recency_days": 30, 
        "ActionsLast30Days": 5, "num_calls": 1
    }])

    try:
        with st.spinner("Processing data..."):
            cleaned_data = clean_data(raw_data)
            final_input = cleaned_data.reindex(columns=feature_schema, fill_value=0)

            prediction = model.predict(final_input)[0]
            proba = model.predict_proba(final_input)[0][1]

            st.markdown("---")
            st.write("### Analysis Result")
            
            res1, res2 = st.columns(2)
            res1.metric("Churn Probability", f"{proba:.2%}")
            
            if prediction == 1:
                res2.error("Result: CHURN RISK")
                st.warning("Recommendation: Contact customer for loyalty offer.")
            else:
                res2.success("Result: STABLE")
                st.info("Customer sentiment appears positive.")
                
    except Exception as e:
        st.error(f"Prediction Error: {e}")