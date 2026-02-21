import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Paths
embeddings_path = "embeddings/embeddings.pickle"
model_path = "models/svm_model.pkl"
le_path = "models/label_encoder.pkl"

# Load embeddings
print("[INFO] Loading embeddings...")

with open(embeddings_path, "rb") as f:
    data = pickle.loads(f.read())

X = data["embeddings"]
y = data["names"]

# Encode labels
print("[INFO] Encoding labels...")

le = LabelEncoder()
labels = le.fit_transform(y)

# Train SVM
print("[INFO] Training SVM model...")

recognizer = SVC(
    C=1.0,
    kernel="linear",
    probability=True
)

recognizer.fit(X, labels)

# Create models folder
if not os.path.exists("models"):
    os.makedirs("models")

# Save model
print("[INFO] Saving model...")

with open(model_path, "wb") as f:
    f.write(pickle.dumps(recognizer))

with open(le_path, "wb") as f:
    f.write(pickle.dumps(le))

print("[INFO] Training completed successfully")