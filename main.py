# import cv2
# from deepface import DeepFace
# import os

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     try:
#         # Perform face recognition
#         result = DeepFace.find(frame, db_path="faces_db", model_name="VGG-Face")

#         if result and not result[0].empty:
#             # Extract recognized person's name from the image path
#             matched_image_path = result[0]["identity"][0]
#             person_name = os.path.basename(matched_image_path).split(".")[0]

#             # Display the person's name on the screen
#             cv2.putText(frame, f"Recognized: {person_name}", (50, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "Unknown Face", (50, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     except Exception as e:
#         print("Error:", e)

#     # Show the video frame
#     cv2.imshow("Face Recognition", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
