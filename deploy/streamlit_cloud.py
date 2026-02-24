import streamlit as st
import pandas as pd
import joblib
import json
import os
import sys
from PIL import Image

# 1. PAGE CONFIG (This adds the Emoji to your Browser Tab)
st.set_page_config(
    page_title="Reder Churn Prediction",
    page_icon="📡", # Your new emoji icon
    layout="wide"
)

# 1. PATH FIX
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

try:
    from DataCleaning.clean import clean_data
except ModuleNotFoundError:
    st.error("Module 'DataCleaning' not found.")

# --- CUSTOM CSS FOR SLIM BUTTON & CLEAN TEXT ---
st.markdown("""
    <style>
    /* Ultra-Slim Banner */
    .slim-banner {
        background-color: #4A90E2;
        padding: 5px 0px;
        border-radius: 4px;
        margin-bottom: 10px;
        width: 100%;
    }
    .banner-text {
        color: white;
        text-align: center;
        margin: 0;
        font-family: sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
    }
    
    /* Make the Predict Button transparent and small */
    div.stButton > button {
        background-color: transparent !important;
        color: #4A90E2 !important;
        border: 1px solid #4A90E2 !important;
        padding: 5px 20px !important;
        font-weight: 500 !important;
        border-radius: 5px !important;
        display: block;
        margin: 0 auto; /* Centers the button */
    }
    div.stButton > button:hover {
        background-color: #4A90E2 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (LOGO) ---
with st.sidebar:
    try:
        logo_path = os.path.join(current_dir, "radiologo.png") 
        st.image(Image.open(logo_path), use_container_width=True)
    except:
        st.markdown("<h3 style='color: #4A90E2;'>📡 REDER</h3>", unsafe_allow_html=True)

# --- SLIM BOLD BANNER ---
st.markdown(f"""<div class="slim-banner"><p class="banner-text">REDER TELECOM CHURN PREDICTION</p></div>""", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Predictor", "📈 Metrics", "ℹ️ About"])

with tab1:
    st.markdown("### Customer Details")
    
    # LOAD ASSETS
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

    st.markdown("<br>", unsafe_allow_html=True) # Adding a little space
    
    # PREDICTION BUTTON (Now small, centered, and transparent)
    if st.button("Analyze Customer"):
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
                
                # CLEAN STATUS MESSAGES
                if prediction == 1:
                    st.markdown(f"<h3 style='color: #D32F2F; text-align: center;'>🚩 High Churn Risk: {proba:.1%}</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;'>This customer is likely to churn. Intervention recommended.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color: #388E3C; text-align: center;'>✅ Healthy Status: {proba:.1%}</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;'>This customer is stable. No action needed.</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.markdown("### Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "89%")
    col2.metric("Precision", "84%")
    col3.metric("Recall", "81%")

with tab3:
    st.markdown("### System Info")
    st.write("Real-time behavioral churn prediction for Reder Telecom.")