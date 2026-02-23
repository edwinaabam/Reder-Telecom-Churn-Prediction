import streamlit as st
import pandas as pd
import joblib  # You might need 'import pickle' depending on how you saved it
import os
import sys

# 1. NEW: Load the model directly instead of using an API_URL
# Make sure your model file is in a folder named 'models' in your repo
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')

@st.cache_resource  # This keeps the model in memory so it doesn't reload every time
def load_my_model():
    try:
        # If you used joblib to save your model:
        return joblib.load(MODEL_PATH)
    except:
        # Fallback for standard pickle if joblib fails
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)

# Initialize model
model = load_my_model()

st.set_page_config(page_title="Reder Telecom Churn", page_icon="📡")
st.title("Telecom Customer Churn Prediction")

st.header("Customer Details")

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
    # 2. NEW: Convert input into a DataFrame that your model expects
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Plan": plan,
        "NPS": nps,
        "PageViews": page_views,
        "Logins": logins,
        "Rating": rating
    }])

    try:
        with st.spinner("Analyzing customer data..."):
            # 3. NEW: Predict directly using the model instead of requests.post
            prediction_proba = model.predict_proba(input_data)[0][1]
            prediction_label = 1 if prediction_proba > 0.5 else 0
            prediction_class = "Churn" if prediction_label == 1 else "Not Churn"

            st.markdown("---")
            st.subheader("Prediction Result")

            st.metric("Churn Probability", f"{prediction_proba:.2%}")
            
            if prediction_label == 1:
                st.error(f"⚠️ Result: {prediction_class}")
                st.warning("Recommendation: Proactive retention call recommended.")
            else:
                st.success(f"✅ Result: {prediction_class}")
                st.info("Customer is stable.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure your model expects the feature names: Age, Gender, Plan, etc.")