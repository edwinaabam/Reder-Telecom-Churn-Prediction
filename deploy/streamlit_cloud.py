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

# --- CUSTOM CSS FOR LIGHTER THEME ---
st.markdown("""
    <style>
    /* Styled Sidebar buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #4A90E2;
        background-color: white;
        color: #4A90E2;
        font-weight: 500;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4A90E2;
        color: white;
    }
    
    /* Lighter Steel Blue Banner */
    .custom-banner {
        background-color: #4A90E2; /* Lighter Blue */
        padding: 6px;
        border-radius: 6px;
        margin-bottom: 25px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    # LOGO
    try:
        logo_path = os.path.join(current_dir, "telecomlogo.png") 
        st.image(Image.open(logo_path), use_container_width=True)
    except:
        st.markdown("<h2 style='text-align: center; color: #4A90E2;'>📡 Reder</h2>", unsafe_allow_html=True)

    st.markdown("---")
    
    if "page" not in st.session_state:
        st.session_state.page = "Churn Predictor"

    # Clean Menu Buttons
    if st.button("📊 Churn Predictor"):
        st.session_state.page = "Churn Predictor"
    
    if st.button("📈 Model Metrics"):
        st.session_state.page = "Model Metrics"
        
    if st.button("ℹ️ About this app"):
        st.session_state.page = "About this app"

# --- UNIFORM LIGHTER BANNER ---
st.markdown(f"""
    <div class="custom-banner">
        <h3 style="color:white; text-align:center; margin:0; font-family:sans-serif; font-size:1rem; font-weight:500; letter-spacing: 0.5px;">
            Reder Telecom Customer Churn Prediction
        </h3>
    </div>
    """, unsafe_allow_html=True)

# 2. PAGE CONTENT LOGIC
if st.session_state.page == "Churn Predictor":
    st.markdown("### Customer Behavior Analysis")
    
    # Pathing for model files
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

    # Layout inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        plan = st.selectbox("Plan", ["Basic", "Express", "Premium"])
        nps = st.slider("NPS Score (0-10)", 0, 10, 5)
    with col2:
        page_views = st.number_input("Page Views", 0, 1000, 20)
        logins = st.number_input("Logins", 0, 100, 5)
        rating = st.selectbox("Customer Rating", [1, 2, 3, 4, 5], index=2)

    if st.button("Run Prediction", type="primary"):
        raw_data = pd.DataFrame([{
            "Age": age, "Gender": gender, "Plan": plan, "NPS": nps,
            "PageViews": page_views, "Logins": logins, "Rating": rating,
            "Frequency": "weekly", "Recency_days": 30, 
            "ActionsLast30Days": 5, "num_calls": 1
        }])
        try:
            with st.spinner("Processing..."):
                cleaned_data = clean_data(raw_data)
                final_input = cleaned_data.reindex(columns=feature_schema, fill_value=0)
                prediction = model.predict(final_input)[0]
                proba = model.predict_proba(final_input)[0][1]
                
                st.markdown("---")
                res1, res2 = st.columns(2)
                res1.metric("Risk Score", f"{proba:.2%}")
                if prediction == 1:
                    res2.error("Status: AT RISK")
                else:
                    res2.success("Status: STABLE")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

elif st.session_state.page == "Model Metrics":
    st.markdown("### Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "89%")
    col2.metric("Precision", "84%")
    col3.metric("Recall", "81%")

elif st.session_state.page == "About this app":
    st.markdown("### About the System")
    st.write("This platform is built to provide Reder Telecom with data-driven insights to reduce churn.")
    st.info("System uses Random Forest classification with real-time feature engineering.")