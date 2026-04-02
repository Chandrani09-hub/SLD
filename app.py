import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 64

st.title("Sign Language Detection")

model = load_model('model/sign_language_model.h5')

with open('model/class_names.txt') as f:
    class_names = f.read().splitlines()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {label}")
