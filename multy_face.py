import cv2
import json
import numpy as np

from deepface import DeepFace

# Load precomputed embeddings
with open("face_embeddings.json", "r") as f:
    known_embeddings = json.load(f)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Crop face

        try:
            # Get face embedding
            live_embedding = DeepFace.represent(face_img, model_name="VGG-Face")[0]["embedding"]

            # Compare with known embeddings
            best_match = "Unknown"
            min_distance = float("inf")
            threshold = 10  # Adjust if needed

            for person, stored_embedding in known_embeddings.items():
                distance = np.linalg.norm(np.array(live_embedding) - np.array(stored_embedding))
                if distance < min_distance:
                    min_distance = distance
                    best_match = person

            # If the match is confident, display the name
            text = best_match if min_distance < threshold else "Unknown"

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print("Error:", e)

    # Show frame with recognized names
    cv2.imshow("Multi-Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
