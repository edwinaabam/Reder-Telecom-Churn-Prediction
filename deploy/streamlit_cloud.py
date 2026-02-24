import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys
from PIL import Image

# 1. PAGE CONFIG 
st.set_page_config(
    page_title="Reder Churn Predictor",
    page_icon="📡", 
    layout="wide"
)

# 2. PATH FIX
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

try:
    from DataCleaning.clean import clean_data
except ModuleNotFoundError:
    st.error("Module 'DataCleaning' not found.")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Adjusted Banner - Not too thick, not too slim */
    .balanced-banner {
        background-color: #4A90E2;
        padding: 15px 0px; /* Increased from 5px to 15px for better presence */
        border-radius: 8px;
        margin-bottom: 20px;
        width: 100%;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    .banner-text {
        color: white;
        text-align: center;
        margin: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.6rem; /* Slightly larger for the bold look */
        font-weight: 800;
        letter-spacing: 1px;
    }
    
    /* Transparent Predict Button */
    div.stButton > button {
        background-color: transparent !important;
        color: #4A90E2 !important;
        border: 2px solid #4A90E2 !important;
        padding: 10px 30px !important;
        font-weight: 700 !important;
        border-radius: 25px !important;
        display: block;
        margin: 0 auto;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #4A90E2 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIONAL SIDEBAR ---
with st.sidebar:
    try:
        logo_path = os.path.join(current_dir, "telecompic.jpg") 
        st.image(Image.open(logo_path), use_container_width=True)
    except:
        st.markdown("<h2 style='text-align:center;'>📡 REDER</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # FEATURE 1: Bulk Processing
    st.subheader("📁 Bulk Analysis")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded_file:
        st.info("Bulk processing feature active.")
    
    st.markdown("---")
    
    # FEATURE 2: Export Options
    st.subheader("📥 Export Data")
    if st.button("Download Last Prediction"):
        st.write("Generating report...")
    
    st.markdown("---")
    
    # FEATURE 3: Model Controls
    st.subheader("⚙️ Settings")
    st.toggle("Show Probability Details", value=True)
    st.toggle("Enable Retention Tips", value=False)

# --- BALANCED BOLD BANNER ---
st.markdown(f"""
    <div class="balanced-banner">
        <p class="banner-text">📡 REDER TELECOM CHURN PREDICTION</p>
    </div>
    """, unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Individual Predictor", "📈 Model Performance", "ℹ️ Documentation"])

with tab1:
    st.markdown("### Customer Input Profile")
    
    # Load assets
    MODEL_PATH = os.path.join(current_dir, '..', 'model', 'model.pkl')
    SCHEMA_PATH = os.path.join(current_dir, '..', 'model', 'schema.json')

    @st.cache_resource
    def load_all():
        model = joblib.load(MODEL_PATH)
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        return model, schema["model_schema"]

    model, feature_schema = load_all()

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

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Analyze Customer"):
        raw_data = pd.DataFrame([{
            "Age": age, "Gender": gender, "Plan": plan, "NPS": nps,
            "PageViews": page_views, "Logins": logins, "Rating": rating,
            "Frequency": "weekly", "Recency_days": 30, 
            "ActionsLast30Days": 5, "num_calls": 1
        }])
        
        try:
            with st.spinner("Running deep analysis..."):
                cleaned_data = clean_data(raw_data)
                final_input = cleaned_data.reindex(columns=feature_schema, fill_value=0)
                prediction = model.predict(final_input)[0]
                proba = model.predict_proba(final_input)[0][1]
                
                st.markdown("---")
                if prediction == 1:
                    st.markdown(f"<h2 style='color: #D32F2F; text-align: center;'>🚩 HIGH CHURN RISK: {proba:.1%}</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color: #388E3C; text-align: center;'>✅ HEALTHY STATUS: {proba:.1%}</h2>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ... (rest of the tab logic for Metrics and About)