import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 64

st.title("Sign Language Detection")

# Load model
model = load_model("model/sign_language_model.h5")

# Load class names
with open("model/class_names.txt") as f:
    class_names = f.read().splitlines()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to OpenCV format
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]

    st.success(f"Predicted Sign: {label}")
