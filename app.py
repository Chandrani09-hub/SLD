import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

st.title("Sign Language Detection App")

# Load model
model_path = "model/model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
else:
    st.warning("Model file not found.")

# Upload image
uploaded_file = st.file_uploader("Upload a sign language image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    # Example: convert to numpy array
    img_array = np.array(img)
    st.write("Image shape:", img_array.shape)
