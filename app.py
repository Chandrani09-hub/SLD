import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Constants
IMG_SIZE = 64

# Title
st.title(" Sign Language Detection App")

# Load model
@st.cache_resource
def load_my_model():
    return load_model('model/sign_language_model.h5')

model = load_my_model()

# Load class names
with open('model/class_names.txt') as f:
    class_names = f.read().splitlines()

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    label = class_names[predicted_class]
    confidence = np.max(prediction)

    # Output
    st.success(f"✅ Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}")
