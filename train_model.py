import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import random

# ==============================
# SET SEEDS FOR REPRODUCIBILITY
# ==============================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==============================
# PATH TO TRAINING DATA
# ==============================
DATA_DIR = "C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/data/asl_alphabet_train"
IMG_SIZE = 64  # resize all images to 64x64

# ==============================
# LOAD DATA
# ==============================
def load_data(limit_per_class=50):
    images, labels = [], []
    classes = sorted(os.listdir(DATA_DIR))

    for idx, cls in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(class_dir):
            continue

        # Limit number of images per class
        for img_name in os.listdir(class_dir)[:limit_per_class]:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)

    return np.array(images), np.array(labels), classes

print("Loading data...")
X, y, classes = load_data(limit_per_class=50)
X = X / 255.0  # normalize
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ==============================
# BUILD MODEL
# ==============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==============================
# TRAIN MODEL
# ==============================
print("Training model...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# ==============================
# SAVE MODEL AND CLASS NAMES
# ==============================
# Ensure model folder exists (relative to src/)
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
os.makedirs(model_dir, exist_ok=True)

# Save Keras model
model_path = os.path.join(model_dir, 'model.h5')
model.save(model_path)

# Save class names
class_names_path = os.path.join(model_dir, 'class_names.txt')
with open(class_names_path, "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print(f"✅ Training complete and model saved at {model_path}")
