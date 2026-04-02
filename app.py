import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale using OpenCV
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    st.image(gray, caption="Grayscale Image", use_column_width=True)

    # Dummy prediction (replace later with ML model)
    st.write("Prediction: Demo Output ✅")
