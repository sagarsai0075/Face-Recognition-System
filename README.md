# Face Recognition System

Simple face recognition project using OpenCV DNN + OpenFace embeddings + SVM classifier.

This repository supports two ways to use the system:
- **CLI pipeline** (dataset collection, training, and live recognition)
- **Streamlit app** (camera-based training and recognition from browser UI)

## Features

- Face detection with Caffe SSD (`res10_300x300_ssd`)
- Face embedding extraction with OpenFace (`nn4.small2.v1`)
- Classification with `SVC` (linear kernel)
- Single-person fallback classifier support (`DummyClassifier`)
- Interactive Streamlit workflow for capture + recognize

## Project Structure

```text
Face-Recognition-System/
├── main.py                       # Orchestrates CLI pipeline
├── streamlit_app.py              # Streamlit app
├── requirements.txt
├── runtime.txt                   # python-3.11
├── caffe_model/
├── openface_model/
├── scripts/
│   ├── collect_dataset.py
│   ├── extract_embeddings.py
│   ├── train_svm.py
│   └── recognize.py
├── dataset/                      # Captured training images
├── embeddings/                   # Generated embedding vectors
└── models/                       # Trained model + label encoder
```

## Requirements

- Python **3.11** (as in `runtime.txt`)
- Webcam
- Windows / macOS / Linux (commands below include PowerShell + bash)

## Installation

### 1) Clone and enter project

```powershell
git clone <your-repo-url>
cd Face-Recognition-System
```

### 2) Create virtual environment

**PowerShell (Windows):**

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

**bash (macOS/Linux):**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## CLI Workflow

### Option A: Run full guided flow

```bash
python main.py
```

`main.py` will ask whether to:
1. Capture training images
2. Extract embeddings
3. Train SVM model
4. Start recognition

### Option B: Run each step manually

1) Collect dataset for a person:

```bash
python scripts/collect_dataset.py --name "YourName"
```

2) Extract embeddings:

```bash
python scripts/extract_embeddings.py
```

3) Train classifier:

```bash
python scripts/train_svm.py
```

4) Start live recognition:

```bash
python scripts/recognize.py
```

## Streamlit App Workflow

Run:

```bash
streamlit run streamlit_app.py
```

In the app:
1. Enter a name and capture image for training
2. Click **Capture for Training** (this also rebuilds embeddings and model)
3. Capture another image and click **Recognize**

## Output Files

After training, these files are created/updated:

- `embeddings/embeddings.pickle`
- `models/svm_model.pkl`
- `models/label_encoder.pkl`

Training images are saved under:

- `dataset/<person_name>/`

## Controls

- In OpenCV windows (`collect_dataset.py`, `recognize.py`), press **Q** to quit.

## Common Issues

- **Camera not opening**: close other apps using webcam and retry.
- **No face detected**: use better lighting and keep face clearly visible.
- **Model files missing**: run dataset collection → embeddings extraction → training first.
- **Only one person in dataset**: fallback classifier is used; recognition is limited.

## Dependencies

From `requirements.txt`:

- `opencv-python-headless==4.10.0.84`
- `imutils==0.5.4`
- `scikit-learn==1.5.2`
- `numpy==2.1.3`
- `streamlit==1.39.0`

---

If you want, I can also generate a cleaner **demo section with screenshots/GIF placeholders** for GitHub presentation.
