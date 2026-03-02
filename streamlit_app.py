from pathlib import Path
import pickle
import time

import cv2
import numpy as np
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
MODELS_DIR = ROOT_DIR / "models"

DETECTOR_PROTOTXT = ROOT_DIR / "caffe_model" / "deploy.prototxt"
DETECTOR_MODEL = ROOT_DIR / "caffe_model" / "res10_300x300_ssd_iter_140000.caffemodel"
EMBEDDING_MODEL = ROOT_DIR / "openface_model" / "openface_nn4.small2.v1.t7"

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.pickle"
SVM_MODEL_FILE = MODELS_DIR / "svm_model.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"


@st.cache_resource
def load_networks():
    detector = cv2.dnn.readNetFromCaffe(str(DETECTOR_PROTOTXT), str(DETECTOR_MODEL))
    embedder = cv2.dnn.readNetFromTorch(str(EMBEDDING_MODEL))
    return detector, embedder


def bytes_to_bgr(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from camera input.")
    return image


def detect_primary_face(image: np.ndarray, detector, threshold: float = 0.4):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    detector.setInput(blob)
    detections = detector.forward()

    if detections.shape[2] == 0:
        return None, None

    scores = detections[0, 0, :, 2]
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index])
    if confidence < threshold:
        return None, None

    box = detections[0, 0, best_index, 3:7] * np.array([w, h, w, h])
    (x1, y1, x2, y2) = box.astype("int")

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None, None

    return face, (x1, y1, x2, y2)


def face_embedding(face: np.ndarray, embedder) -> np.ndarray:
    face_blob = cv2.dnn.blobFromImage(
        face,
        1.0 / 255,
        (96, 96),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )
    embedder.setInput(face_blob)
    return embedder.forward().flatten()


def save_training_face(person_name: str, image_bgr: np.ndarray, detector) -> Path:
    person_dir = DATASET_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    face, _ = detect_primary_face(image_bgr, detector, threshold=0.35)
    if face is None:
        raise ValueError("No clear face detected. Please capture again.")

    file_name = f"img_{int(time.time() * 1000)}.jpg"
    output_path = person_dir / file_name
    saved = cv2.imwrite(str(output_path), face)
    if not saved:
        raise RuntimeError("Could not save captured face image.")

    return output_path


def rebuild_embeddings_and_model(detector, embedder):
    if not DATASET_DIR.exists():
        raise ValueError("Dataset folder does not exist yet.")

    embeddings = []
    names = []

    for person_dir in DATASET_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        for image_path in person_dir.iterdir():
            if not image_path.is_file():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            face, _ = detect_primary_face(image, detector, threshold=0.3)
            if face is None:
                continue

            vec = face_embedding(face, embedder)
            embeddings.append(vec)
            names.append(person_name)

    if not embeddings:
        raise ValueError("No valid face embeddings found. Add clearer training images.")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EMBEDDINGS_FILE, "wb") as file:
        pickle.dump({"embeddings": embeddings, "names": names}, file)

    le = LabelEncoder()
    labels = le.fit_transform(names)

    if len(le.classes_) < 2:
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        classifier = SVC(C=1.0, kernel="linear", probability=True)

    classifier.fit(embeddings, labels)

    with open(SVM_MODEL_FILE, "wb") as file:
        pickle.dump(classifier, file)

    with open(LABEL_ENCODER_FILE, "wb") as file:
        pickle.dump(le, file)


def load_recognition_artifacts():
    if not SVM_MODEL_FILE.exists() or not LABEL_ENCODER_FILE.exists():
        raise FileNotFoundError("Trained model not found. Complete Step 1 training first.")

    with open(SVM_MODEL_FILE, "rb") as file:
        recognizer = pickle.load(file)

    with open(LABEL_ENCODER_FILE, "rb") as file:
        label_encoder = pickle.load(file)

    return recognizer, label_encoder


def recognize_image(image_bgr: np.ndarray, detector, embedder):
    recognizer, label_encoder = load_recognition_artifacts()

    face, box = detect_primary_face(image_bgr, detector, threshold=0.4)
    if face is None:
        raise ValueError("No face detected. Please capture again.")

    vec = face_embedding(face, embedder).reshape(1, -1)

    probabilities = recognizer.predict_proba(vec)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    name = label_encoder.classes_[best_index]

    return name, confidence, box


def main():
    st.set_page_config(page_title="Face Recognition", page_icon="📷")
    st.title("Face Recognition System")

    detector, embedder = load_networks()

    st.header("Step 1: Capture")
    person_name = st.text_input("Enter name", placeholder="e.g. Rahul").strip()
    train_capture = st.camera_input("Capture image for training", key="train_capture")

    if st.button("Capture for Training", type="primary"):
        if not person_name:
            st.error("Please enter a name first.")
        elif train_capture is None:
            st.error("Please capture an image first.")
        else:
            try:
                image_bgr = bytes_to_bgr(train_capture)
                output_path = save_training_face(person_name, image_bgr, detector)
                with st.spinner("Updating model..."):
                    rebuild_embeddings_and_model(detector, embedder)
                st.success(f"Saved and trained successfully: {output_path.name}")
            except Exception as exc:
                st.error(str(exc))

    st.header("Step 2: Recognize")
    recognize_capture = st.camera_input("Capture image for recognition", key="recognize_capture")

    if st.button("Recognize"):
        if recognize_capture is None:
            st.error("Please capture an image first.")
        else:
            try:
                image_bgr = bytes_to_bgr(recognize_capture)
                name, confidence, _ = recognize_image(image_bgr, detector, embedder)
                st.success(f"Recognized: {name} ({confidence:.2%})")
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
