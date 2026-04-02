import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd

# -------------------------
# Load your pickle model
# -------------------------
MODEL_PATH = "model.pkl"  # make sure your model file is in the repo root

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully ✅")
else:
    st.error(f"Model file not found at {MODEL_PATH} ❌")
    st.stop()  # Stop execution if model is missing

# -------------------------
# Example prediction
# -------------------------
def predict(input_data):
    # Make sure input_data is in the same format your model expects
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    return prediction

# -------------------------
# Streamlit UI
# -------------------------
st.title("My ML App")
user_input = st.text_input("Enter input values separated by commas", "0,0,0")

if st.button("Predict"):
    try:
        # Convert input string to list of floats
        input_list = [float(x.strip()) for x in user_input.split(",")]
        result = predict(input_list)
        st.write(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {e}")
