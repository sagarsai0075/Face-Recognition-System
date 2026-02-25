import cv2
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Collect face images for a person")
    parser.add_argument(
        "--name",
        default=os.environ.get("PERSON_NAME", "Sagar"),
        
    )
    return parser.parse_args()


args = parse_args()
person_name = args.name.strip()

if not person_name:
    raise ValueError("Person name cannot be empty")

# Dataset folder
dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)

# Create folder
if not os.path.exists(person_path):
    os.makedirs(person_path)

# Load Caffe face detector
prototxt = "caffe_model/deploy.prototxt"
model = "caffe_model/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
max_images = 100

print("[INFO] Camera started. Look at camera and rotate face slowly.")
print("[INFO] Press Q to quit")

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

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:
           
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure bounding box is within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                print("[DEBUG] Empty face region, skipping.")
                continue

            count += 1
            filename = f"{person_path}/img_{count}.jpg"
            cv2.imwrite(filename, face)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Saved: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Collect Dataset", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if count >= max_images:
        break


cap.release()
cv2.destroyAllWindows()
print("[INFO] Dataset collection finished")