import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

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

if len(X) == 0 or len(y) == 0:
    raise ValueError("No embeddings found. Please collect dataset and extract embeddings first.")

# Encode labels
print("[INFO] Encoding labels...")

le = LabelEncoder()
labels = le.fit_transform(y)

# Train SVM
print("[INFO] Training SVM model...")

if len(le.classes_) < 2:
    print("[WARN] Only one class found. Using fallback classifier for single-person recognition.")
    recognizer = DummyClassifier(strategy="most_frequent")
else:
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