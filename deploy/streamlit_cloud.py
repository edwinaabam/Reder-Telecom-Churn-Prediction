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

# --- CUSTOM CSS FOR CLEAN LINKS & BOLD BANNER ---
st.markdown("""
    <style>
    /* 1. Remove Button Styling to make them look like simple list items */
    div.stButton > button {
        border: none !important;
        background-color: transparent !important;
        color: #4A90E2 !important;
        text-align: left !important;
        padding: 0px !important;
        font-size: 18px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        color: #002b5b !important;
        text-decoration: underline !important;
    }
    
    /* 2. Bold and Bigger Banner Title */
    .custom-banner {
        background-color: #4A90E2;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 25px;
        width: 100%;
    }
    .banner-text {
        color: white;
        text-align: center;
        margin: 0;
        font-family: sans-serif;
        font-size: 1.8rem; /* Bigger font */
        font-weight: 800;   /* Extra Bold */
        text-transform: uppercase; /* Makes it punchy */
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    # LOGO
    try:
        logo_path = os.path.join(current_dir, "logo.png") 
        st.image(Image.open(logo_path), use_container_width=True)
    except:
        st.markdown("<h2 style='color: #4A90E2;'>📡 Reder</h2>", unsafe_allow_html=True)

    st.markdown("### Navigation")
    
    if "page" not in st.session_state:
        st.session_state.page = "Churn Predictor"

    # These now look like clean list items instead of buttons
    if st.button("📊 Churn Predictor"):
        st.session_state.page = "Churn Predictor"
    
    if st.button("📈 Model Metrics"):
        st.session_state.page = "Model Metrics"
        
    if st.button("ℹ️ About this app"):
        st.session_state.page = "About this app"

# --- BIG BOLD UNIFORM BANNER ---
st.markdown(f"""
    <div class="custom-banner">
        <h1 class="banner-text">REDER TELECOM CHURN PREDICTION</h1>
    </div>
    """, unsafe_allow_html=True)

# 2. PAGE CONTENT LOGIC
if st.session_state.page == "Churn Predictor":
    st.markdown("### Customer Analysis")
    
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
        nps = st.slider("NPS Score", 0, 10, 5)
    with col2:
        page_views = st.number_input("Page Views", 0, 1000, 20)
        logins = st.number_input("Logins", 0, 100, 5)
        rating = st.selectbox("Customer Rating", [1, 2, 3, 4, 5], index=2)

    if st.button("🚀 Run Prediction", type="primary"):
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

elif st.session_state.page == "Model Metrics":
    st.markdown("### Model Performance")
    # Content...

elif st.session_state.page == "About this app":
    st.markdown("### About the System")
    # Content...