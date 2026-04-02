import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64

model = load_model(
    "C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/model/model.h5"
)

with open(
    "C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/model/class_names.txt"
) as f:
    class_names = f.read().splitlines()

img = cv2.imread("test.jpg")  # change image name

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
label = class_names[np.argmax(prediction)]

print("Predicted:", label)
