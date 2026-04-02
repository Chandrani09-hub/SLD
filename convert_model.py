import tensorflow as tf
import joblib
import numpy as np

# Load the trained Keras model (absolute path)
model = tf.keras.models.load_model(
    "C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/model/model.h5"
)

# Wrapper class for sklearn-style prediction
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X = np.array(X)
        X = X.reshape(-1, 64, 64, 3)
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

wrapped_model = ModelWrapper(model)

# Save as .pkl in the same model folder
joblib.dump(
    wrapped_model,
    "C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/model/model.pkl"
)

print("✅ model.pkl created successfully!")