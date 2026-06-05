import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟"
)

st.title("🤟 Sign Language Detection App")

# ==========================
# LOAD MODEL
# ==========================
model_path = os.path.join("model", "model.h5")

if not os.path.exists(model_path):
    st.error("❌ model.h5 not found in model folder.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================
# LOAD CLASS NAMES
# ==========================
class_names_path = os.path.join("model", "class_names.txt")

if not os.path.exists(class_names_path):
    st.error("❌ class_names.txt not found in model folder.")
    st.stop()

with open(class_names_path, "r") as f:
    class_names = f.read().splitlines()

# ==========================
# FILE UPLOADER
# ==========================
uploaded_file = st.file_uploader(
    "Upload a Sign Language Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# PREDICTION
# ==========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((64, 64))

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array.astype("float32") / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Sign: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")
