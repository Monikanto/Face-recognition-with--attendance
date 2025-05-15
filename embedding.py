import os
import json
from deepface import DeepFace

db_path = "faces_db"
embeddings = {}

# Loop through each image and compute embeddings
for img_name in os.listdir(db_path):
    img_path = os.path.join(db_path, img_name)
    try:
        embedding = DeepFace.represent(img_path, model_name="VGG-Face")[0]["embedding"]
        person_name = os.path.splitext(img_name)[0]  # Remove .jpg
        embeddings[person_name] = embedding
    except Exception as e:
        print(f"Skipping {img_name}: {e}")

# Save embeddings to a JSON file
with open("face_embeddings.json", "w") as f:
    json.dump(embeddings, f)

print("Face embeddings saved!")
