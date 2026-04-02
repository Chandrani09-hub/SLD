import cv2
import os

label = input("Enter label: ")

save_dir = f"C:/Users/HP/OneDrive/Desktop/sign-language-detection/sign-language-detection/data/{label}"

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        path = os.path.join(save_dir, f"{label}_{count}.jpg")
        cv2.imwrite(path, frame)
        print("Saved:", path)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
