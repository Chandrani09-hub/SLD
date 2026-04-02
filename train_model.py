import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==============================
# PATH TO TRAINING DATA
# ==============================
DATA_DIR = "C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/data/asl_alphabet_train"
IMG_SIZE = 64  # resize all images to 64x64

# ==============================
# LOAD DATA
# ==============================
def load_data():
    images, labels = [], []
    classes = sorted(os.listdir(DATA_DIR))

    for idx, cls in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(class_dir):
            continue

        # Limit to 50 images per class to avoid memory issues (adjust as needed)
        for img_name in os.listdir(class_dir)[:50]:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)

    return np.array(images), np.array(labels), classes

print("Loading data...")
X, y, classes = load_data()
X = X / 255.0
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
# Make sure model folder exists outside src
os.makedirs("../model", exist_ok=True)

# Save Keras model
model.save("../model/model.h5")

# Save class names
with open("../model/class_names.txt", "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print("✅ Training complete and model saved in ../model/model.h5")
