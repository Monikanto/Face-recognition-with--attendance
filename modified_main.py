# import cv2
# import json
# import numpy as np
# from deepface import DeepFace

# # Load precomputed embeddings
# with open("face_embeddings.json", "r") as f:
#     known_embeddings = json.load(f)

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     try:
#         # Get face embedding from live frame
#         live_embedding = DeepFace.represent(frame, model_name="VGG-Face")[0]["embedding"]

#         # Compare live embedding with known embeddings
#         best_match = None
#         min_distance = float("inf")

#         for person, stored_embedding in known_embeddings.items():
#             distance = np.linalg.norm(np.array(live_embedding) - np.array(stored_embedding))
#             if distance < min_distance:
#                 min_distance = distance
#                 best_match = person

#         # Set a threshold for recognition (adjustable)
#         threshold = 10  # Lower value = stricter match
#         if min_distance < threshold:
#             cv2.putText(frame, f"Recognized: {best_match}", (50, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "Unknown Face", (50, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     except Exception as e:
#         print("Error:", e)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
