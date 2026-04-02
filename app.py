import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

st.title("Sign Language Detection App")

# Load the pickled model
model = joblib.load("model/model.pkl")

# Load class names
with open("model/class_names.txt") as f:
    class_names = f.read().splitlines()

uploaded_file = st.file_uploader("Upload a sign language image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img_resized = cv2.resize(img_array, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction_index = model.predict(img_input)[0]
    predicted_label = class_names[prediction_index]

    st.write(f"Prediction: **{predicted_label}**")
