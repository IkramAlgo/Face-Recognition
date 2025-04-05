import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load embeddings
with open("encodings.pkl", "rb") as f:
    embeddings, labels = pickle.load(f)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train SVM
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(embeddings, labels_encoded)

# Save the model and label encoder
with open("model.pkl", "wb") as f:
    pickle.dump((clf, label_encoder), f)

print("✅ Model training complete!")
