import cv2
import pickle
import numpy as np

# Paths
detector_prototxt = "caffe_model/deploy.prototxt"
detector_model = "caffe_model/res10_300x300_ssd_iter_140000.caffemodel"
embedding_model = "openface_model/openface_nn4.small2.v1.t7"
svm_model_path = "models/svm_model.pkl"
le_path = "models/label_encoder.pkl"

# Load models
print("[INFO] Loading models...")

detector = cv2.dnn.readNetFromCaffe(detector_prototxt, detector_model)
embedder = cv2.dnn.readNetFromTorch(embedding_model)

with open(svm_model_path, "rb") as f:
    recognizer = pickle.load(f)

with open(le_path, "rb") as f:
    le = pickle.load(f)

# Start camera
cap = cv2.VideoCapture(0)

print("[INFO] Starting face recognition. Press Q to quit.")

while True:

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not working")
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]

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

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Show result
            text = f"{name}: {proba:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()