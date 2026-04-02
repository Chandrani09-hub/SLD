import cv2
import os

label = input("Enter label for the sign: ")
save_dir = f"data/{label}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
count = 0

print("Press 'c' to capture images. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture - Press 'c' to save image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        img_path = os.path.join(save_dir, f"{label}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
