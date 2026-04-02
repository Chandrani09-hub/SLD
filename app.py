import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64

print("Loading model...")
model = load_model('model/sign_language_model.h5')

with open('model/class_names.txt') as f:
    class_names = f.read().splitlines()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    x0, y0, width = 100, 100, 200
    roi = frame[y0:y0+width, x0:x0+width]

    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]

    cv2.rectangle(frame, (x0, y0), (x0+width, y0+width), (255, 0, 0), 2)
    cv2.putText(frame, f'Prediction: {label}', (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
