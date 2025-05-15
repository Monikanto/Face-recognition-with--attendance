import cv2
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace


# Load precomputed face embeddings
with open("face_embeddings.json", "r") as f:
    known_embeddings = json.load(f)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize attendance log
attendance_file = "attendance.csv"

# Function to mark attendance
def mark_attendance(name):
    df = pd.read_csv(attendance_file) if os.path.exists(attendance_file) else pd.DataFrame(columns=["Name", "Date", "Time"])
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if attendance is already marked today
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_entry = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Attendance Marked: {name}")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Crop face

        try:
            # Get face embedding
            live_embedding = DeepFace.represent(face_img, model_name="VGG-Face")[0]["embedding"]

            # Compare with stored embeddings
            best_match = "Unknown"
            min_distance = float("inf")
            threshold = 10  # Adjust if needed

            for person, stored_embedding in known_embeddings.items():
                distance = np.linalg.norm(np.array(live_embedding) - np.array(stored_embedding))
                if distance < min_distance:
                    min_distance = distance
                    best_match = person

            # Display the recognized name
            text = best_match if min_distance < threshold else "Unknown"
            color = (0, 255, 0) if text != "Unknown" else (0, 0, 255)

            # Mark attendance if recognized
            if text != "Unknown":
                mark_attendance(text)

            # Draw face rectangle and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Error:", e)

    # Show the video feed
    cv2.imshow("Face Recognition Attendance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
