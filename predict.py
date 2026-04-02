import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

IMG_SIZE = 64

def predict_image(image_path):
    model = load_model('model/sign_language_model.h5')

    with open('model/class_names.txt') as f:
        class_names = f.read().splitlines()

    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]
    print(f"Predicted Sign: {label}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
