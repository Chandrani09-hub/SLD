import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

DATA_DIR = 'C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/data/asl_alphabet_train' 
IMG_SIZE = 64

def load_data():
    images, labels = [], []
    classes = sorted(os.listdir(DATA_DIR))

    for idx, cls in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, cls)
        for img_name in os.listdir(class_dir)[:500]: 
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)

    return np.array(images), np.array(labels), classes

print("Loading data...")
X, y, class_names = load_data()

X = X / 255.0  
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

print("Saving model...")
os.makedirs('model', exist_ok=True)
model.save('model/sign_language_model.h5')

with open('model/class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))

print("Training complete and model saved!")
