import streamlit as st
import numpy as np
import joblib
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="Sign Language Detection", page_icon="🤟")
st.title("🤟 Sign Language Detection App")
st.write("Upload a hand sign image and the app will detect the letter!")

@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    with open("model/class_names.txt") as f:
        class_names = f.read().splitlines()
    return model, class_names

model, class_names = load_model()

mp_hands = mp.solutions.hands

def extract_landmarks(image_rgb):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            data = []
            for lm in results.multi_hand_landmarks[0].landmark:
                data.append(lm.x)
                data.append(lm.y)
            return np.array(data)
    return None

uploaded_file = st.file_uploader("Upload a hand sign image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting hand sign..."):
        features = extract_landmarks(img_array)

    if features is not None:
        features = features.reshape(1, -1)
        try:
            pred_index = model.predict(features)[0]
            label = class_names[pred_index] if isinstance(pred_index, int) else pred_index
            st.success(f"### Detected Sign: **{label}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("⚠️ No hand detected. Please try a clearer image.")
