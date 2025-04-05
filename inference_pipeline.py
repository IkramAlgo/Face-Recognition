import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Load model and label encoder
with open("model.pkl", "rb") as f:
    clf, label_encoder = pickle.load(f)

# Start webcam
cap = cv2.VideoCapture(0)

print("📸 Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Try to detect and recognize face
        result = DeepFace.represent(frame, model_name='Facenet')[0]
        embedding = result["embedding"]
        
        # Predict with SVM
        pred = clf.predict([embedding])
        prob = clf.predict_proba([embedding])[0]
        name = label_encoder.inverse_transform(pred)[0]
        confidence = np.max(prob) * 100

        # Show prediction
        cv2.putText(frame, f"{name} ({confidence:.2f}%)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except:
        cv2.putText(frame, "Face Not Recognized", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed.")
# This script captures video from the webcam, detects faces, and recognizes them using the trained SVM model.
# It displays the recognized name and confidence on the video feed. The script runs until 'q' is pressed.
# Make sure to have the webcam connected and permissions granted for video capture.