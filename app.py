import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

st.title("Sign Language Detection App")

# Load model
model = joblib.load("model/model.pkl")

# Load class names
with open("model/class_names.txt") as f:
    class_names = f.read().splitlines()

uploaded_file = st.file_uploader("Upload a sign language image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image and convert to array
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img_array, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    # Predict
    try:
        prediction_index = np.argmax(model.predict(img_input), axis=1)[0]
        predicted_label = class_names[prediction_index]
        st.write(f"Prediction: **{predicted_label}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
