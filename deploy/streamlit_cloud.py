import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys
from PIL import Image

# 1. PATH FIX
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

try:
    from DataCleaning.clean import clean_data
except ModuleNotFoundError:
    st.error("Module 'DataCleaning' not found.")

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    # LOGO
    try:
        logo_path = os.path.join(current_dir, "logo.png") 
        st.image(Image.open(logo_path), use_container_width=True)
    except:
        st.info("📡 Reder Telecom")

    st.markdown("---")
    st.subheader("Navigation")
    
    # Listed out buttons
    if "page" not in st.session_state:
        st.session_state.page = "Churn Predictor"

    if st.button("📊 Churn Predictor", use_container_width=True):
        st.session_state.page = "Churn Predictor"
    
    if st.button("📈 Model Metrics", use_container_width=True):
        st.session_state.page = "Model Metrics"
        
    if st.button("ℹ️ About this app", use_container_width=True):
        st.session_state.page = "About this app"

# --- THIN COLORED BANNER (NAVY BLUE) ---
# Reduced padding from 15px to 5px and font-size to 1.2rem for a "thin" look
st.markdown("""
    <div style="background-color:#002b5b; padding:5px; border-radius:8px; margin-bottom:20px; border-left: 8px solid #FF4B4B;">
        <h3 style="color:white; text-align:center; margin:0; font-family:sans-serif; font-size:1.2rem; font-weight:400;">
            Reder Telecom Customer Churn Prediction
        </h3>
    </div>
    """, unsafe_allow_html=True)

# 2. PAGE LOGIC
if st.session_state.page == "Churn Predictor":
    # --- LOAD ASSETS ---
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
        st.error(f"Asset Load Error: {e}")

    st.markdown("#### Customer Data Entry")
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

    if st.button("Calculate Churn Risk", type="primary"):
        raw_data = pd.DataFrame([{
            "Age": age, "Gender": gender, "Plan": plan, "NPS": nps,
            "PageViews": page_views, "Logins": logins, "Rating": rating,
            "Frequency": "weekly", "Recency_days": 30, 
            "ActionsLast30Days": 5, "num_calls": 1
        }])
        try:
            with st.spinner("Analyzing..."):
                cleaned_data = clean_data(raw_data)
                final_input = cleaned_data.reindex(columns=feature_schema, fill_value=0)
                prediction = model.predict(final_input)[0]
                proba = model.predict_proba(final_input)[0][1]
                
                st.markdown("---")
                res1, res2 = st.columns(2)
                res1.metric("Churn Probability", f"{proba:.2%}")
                if prediction == 1:
                    res2.error("Result: HIGH RISK")
                else:
                    res2.success("Result: STABLE")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

elif st.session_state.page == "Model Metrics":
    st.markdown("### Model Performance Metrics")
    st.write("Below are the evaluation scores for the current Random Forest model.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "89%")
    m2.metric("Precision", "84%")
    m3.metric("Recall", "81%")

elif st.session_state.page == "About this app":
    st.markdown("### About the Project")
    st.write("""
    This tool is designed for the retention team to predict customer churn based on behavioral data.
    By identifying high-risk customers, the team can intervene with personalized offers to reduce turnover.
    """)
    st.info("**Tech Stack:** Python, Streamlit, Scikit-Learn, FastAPI (Logic Ported)")