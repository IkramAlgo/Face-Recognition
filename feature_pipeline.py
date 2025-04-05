# First Step; Installed Required Libraries 

'''pip install deepface opencv-python scikit-learn pickle-mixin''' 

# Use this command in the terminal and we also have second option to put them in requirements.txt file and then run the command pip install -r requirements.txt

import os
import cv2
import pickle
from deepface import DeepFace

dataset_path = "dataset"
embeddings = []
labels = []

# Loop through each person
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        
        try:
            embedding = DeepFace.represent(img_path=img_path, model_name='Facenet')[0]["embedding"]
            embeddings.append(embedding)
            labels.append(person_name)
            print(f"✔️ Processed {img_name} for {person_name}")
        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")

# Save features to file
with open("encodings.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)

print("✅ Feature extraction complete!")
