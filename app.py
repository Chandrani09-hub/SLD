import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import os

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Sign Language Detection", page_icon="🖐", layout="centered")
st.title("Sign Language Detection App")

# ------------------------------
# Load the model
# ------------------------------
MODEL_PATH = "model"  # Your model file path

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Make sure 'model' is in the repo and tracked via Git LFS.")
else:
    model = joblib.load(MODEL_PATH)

# ------------------------------
# Load class names
# ------------------------------
CLASS_NAMES_PATH = "model/class_names.txt"  # Adjust if class names are elsewhere
if not os.path.exists(CLASS_NAMES_PATH):
    st.error("Class names file not found!")
else:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = f.read().splitlines()

# ------------------------------
# Upload image
# ------------------------------
uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for model (example: resize to 64x64 and flatten)
    img_array = np.array(image.resize((64, 64)))
    img_array = img_array / 255.0  # normalize if needed
    img_array = img_array.reshape(1, -1)  # flatten if your model expects flat input

    # Predict
    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[prediction[0]]
        st.success(f"Predicted Sign: {predicted_class}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
