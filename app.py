import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import os

# Load model
MODEL_PATH = os.path.join("model", "model.pkl")
model = joblib.load(MODEL_PATH)

# Load class names
CLASS_PATH = os.path.join("model", "class_names.txt")
with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.title("Sign Language Detection")
st.write("Upload an image of a hand gesture, and the model will predict the sign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to array and preprocess
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (64, 64))  # Assuming your model was trained on 64x64
    img_input = img_resized.flatten().reshape(1, -1)  # Flatten for scikit-learn model

    # Prediction
    prediction = model.predict(img_input)
    predicted_class = class_names[prediction[0]]

    st.success(f"Predicted Sign: {predicted_class}")
