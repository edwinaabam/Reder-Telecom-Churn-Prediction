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

# --- CUSTOM CSS FOR SLIM BANNER ---
st.markdown("""
    <style>
    /* Ultra-Slim Banner */
    .slim-banner {
        background-color: #4A90E2;
        padding: 5px 0px; /* Very thin padding */
        border-radius: 4px;
        margin-bottom: 10px;
        width: 100%;
    }
    .banner-text {
        color: white;
        text-align: center;
        margin: 0;
        font-family: sans-serif;
        font-size: 1.3rem; /* Large enough to read, small enough to stay slim */
        font-weight: 800;   /* Bold */
    }
    
    /* Remove sidebar top padding to align logo better */
    [data-testid="stSidebarNav"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (ONLY FOR LOGO) ---
with st.sidebar:
    try:
        logo_path = os.path.join(current_dir, "telecomlogo.png") 
        st.image(Image.open(logo_path), use_container_width=True)
    except:
        st.markdown("<h3 style='color: #4A90E2;'>📡 REDER</h3>", unsafe_allow_html=True)

# --- SLIM BOLD BANNER ---
st.markdown(f"""
    <div class="slim-banner">
        <p class="banner-text">REDER TELECOM CHURN PREDICTION</p>
    </div>
    """, unsafe_allow_html=True)

# --- HORIZONTAL TABS (The Navigation) ---
# This places the menu nicely below the banner
tab1, tab2, tab3 = st.tabs(["📊 Churn Predictor", "📈 Model Metrics", "ℹ️ About App"])

# --- PAGE CONTENT LOGIC ---
with tab1:
    st.markdown("### Customer Analysis")
    
    # LOAD ASSETS
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

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        plan = st.selectbox("Plan", ["Basic", "Express", "Premium"])
        nps = st.slider("NPS Score", 0, 10, 5)
    with col2:
        page_views = st.number_input("Page Views", 0, 1000, 20)
        logins = st.number_input("Logins", 0, 100, 5)
        rating = st.selectbox("Customer Rating", [1, 2, 3, 4, 5], index=2)

    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
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
                res1.metric("Risk Score", f"{proba:.2%}")
                if prediction == 1:
                    res2.error("Status: AT RISK")
                else:
                    res2.success("Status: STABLE")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with tab2:
    st.markdown("### Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "89%")
    m2.metric("Precision", "84%")
    m3.metric("Recall", "81%")

with tab3:
    st.markdown("### About the System")
    st.write("This tool identifies at-risk customers for Reder Telecom using behavior analytics.")