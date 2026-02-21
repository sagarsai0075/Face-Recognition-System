import cv2
import os
import pickle
import numpy as np

# Paths
dataset_path = "dataset"
embeddings_path = "embeddings/embeddings.pickle"

# Models
detector_prototxt = "caffe_model/deploy.prototxt"
detector_model = "caffe_model/res10_300x300_ssd_iter_140000.caffemodel"
embedding_model = "openface_model/openface_nn4.small2.v1.t7"

# Load models
print("[INFO] Loading face detector...")
detector = cv2.dnn.readNetFromCaffe(detector_prototxt, detector_model)

print("[INFO] Loading OpenFace model...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

known_embeddings = []
known_names = []

# Loop over dataset
for person in os.listdir(dataset_path):

    person_dir = os.path.join(dataset_path, person)

    if not os.path.isdir(person_dir):
        continue

    print(f"[INFO] Processing {person}")

    for img_name in os.listdir(person_dir):

        img_path = os.path.join(person_dir, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        detector.setInput(blob)
        detections = detector.forward()

        if detections.shape[2] > 0:

            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > 0.3:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_blob = cv2.dnn.blobFromImage(
                    face,
                    1.0 / 255,
                    (96, 96),
                    (0, 0, 0),
                    swapRB=True,
                    crop=False
                )

                embedder.setInput(face_blob)
                vec = embedder.forward()

                known_embeddings.append(vec.flatten())
                known_names.append(person)

# Create embeddings folder
if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

# Save embeddings
print("[INFO] Saving embeddings...")
data = {
    "embeddings": known_embeddings,
    "names": known_names
}

with open(embeddings_path, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Embeddings extraction completed")