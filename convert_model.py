import tensorflow as tf
import joblib
import numpy as np
import os

# ==============================
# PATHS
# ==============================
model_dir = "C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/model"
h5_model_path = os.path.join(model_dir, "model.h5")
pkl_model_path = os.path.join(model_dir, "model.pkl")

# ==============================
# LOAD KERAS MODEL
# ==============================
model = tf.keras.models.load_model(h5_model_path)
print("✅ model.h5 loaded successfully!")

# ==============================
# WRAPPER FOR SKLEARN-STYLE PREDICTION
# ==============================
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 3:  # if single image with shape (64,64,3)
            X = np.expand_dims(X, axis=0)
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

# Wrap the Keras model
wrapped_model = ModelWrapper(model)

# ==============================
# SAVE AS .PKL
# ==============================
joblib.dump(wrapped_model, pkl_model_path)
print(f"✅ model.pkl created successfully at {pkl_model_path}")
