import cv2
import os

# Create a folder if it doesn't exist
db_path = "faces_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

# Load the webcam
cap = cv2.VideoCapture(0)

name = input("Enter the person's name: ").strip()  # Get name for the image file
img_path = os.path.join(db_path, f"{name}.jpg")

print("Press 's' to capture and save the image.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save the image
        cv2.imwrite(img_path, frame)
        print(f"Image saved as {img_path}")
        break
    elif key == ord('q'):  # Press 'q' to quit without saving
        break

cap.release()
cv2.destroyAllWindows()
