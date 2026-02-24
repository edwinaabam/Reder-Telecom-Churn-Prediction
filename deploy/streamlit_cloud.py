import streamlit as st
import pandas as pd
import joblib
import json
import os
from DataCleaning.clean import clean_data # Import your cleaning logic!

# 1. Load Assets (Just like your FastAPI load_assets)
MODEL_PATH = 'model/model.pkl'
SCHEMA_PATH = 'model/schema.json'

@st.cache_resource
def load_all():
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return model, schema["model_schema"]

model, feature_schema = load_all()

# ... (Keep your Streamlit UI inputs here) ...

if st.button("Predict Churn"):
    # 2. Create the raw data frame with ALL possible fields (or just the ones you have)
    raw_data = pd.DataFrame([{
        "Age": age, "Gender": gender, "Plan": plan, "NPS": nps,
        "PageViews": page_views, "Logins": logins, "Rating": rating,
        # Add defaults for other keys your clean_data might expect
        "Frequency": "weekly", "Recency_days": 0, "ActionsLast30Days": 0
    }])

    try:
        # 3. RUN THE CLEANING (This creates the AVGClickDays, etc.)
        cleaned_data = clean_data(raw_data)

        # 4. REINDEX to match the model's exact expected columns
        # This fixes the "Feature names unseen/missing" error!
        final_input = cleaned_data.reindex(columns=feature_schema, fill_value=0)

        # 5. Predict
        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0][1]

        st.metric("Churn Probability", f"{proba:.2%}")
        # ... (display your success/error messages) ...

    except Exception as e:
        st.error(f"Error: {e}")