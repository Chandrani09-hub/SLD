import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ---- App title ----
st.title("My ML Model Deployment App")

# ---- Load model safely ----
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")  # your trained model
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()

# ---- User input ----
st.sidebar.header("Input Features")
feature1 = st.sidebar.number_input("Feature 1", value=0.0)
feature2 = st.sidebar.number_input("Feature 2", value=0.0)
# Add more inputs as needed

input_data = np.array([[feature1, feature2]])  # reshape for model

# ---- Predict button ----
if st.button("Predict"):
    if model is not None:
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.warning("Model not loaded, cannot predict.")
