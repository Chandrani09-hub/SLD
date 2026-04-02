import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

st.title("Sign Language Detection App")

# Load model
model_path = os.path.join("model", "model.pkl")
if not os.path.exists(model_path):
    st.error("Model file not found at 'model/model.pkl'. Please check path.")
else:
    model = joblib.load(model_path)

# Load class names
class_names_path = os.path.join("model", "class_names.txt")
if not os.path.exists(class_names_path):
    st.error("Class names file not found at 'model/class_names.txt'.")
else:
    with open(class_names_path) as f:
        class_names = f.read().splitlines()

# File uploader
uploaded_file = st.file_uploader("Upload a sign language image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to array for prediction
    img_array = np.array(image.resize((64, 64)))  # resize to your model input
    img_array = img_array.reshape(1, 64, 64, 3)  # adapt shape for model
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"Predicted Sign: {predicted_class}")
