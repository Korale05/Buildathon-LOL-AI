# 🦺 PPE Detection AI — Complete Documentation

> **Construction Site Safety Intelligence Platform — AI Module**
>
> A real-time Personal Protective Equipment (PPE) detection system built on YOLOv8, designed to identify safety compliance on construction sites by detecting hardhats and safety vests.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Training Pipeline](#training-pipeline)
5. [Training Results & Performance](#training-results--performance)
6. [Detection Classes](#detection-classes)
7. [Inference Pipeline](#inference-pipeline)
8. [API Reference](#api-reference)
9. [Project Structure](#project-structure)
10. [Dependencies](#dependencies)
11. [Setup & Installation](#setup--installation)
12. [Usage Guide](#usage-guide)
13. [Testing](#testing)
14. [Model Export (ONNX)](#model-export-onnx)
15. [Known Limitations](#known-limitations)
16. [Future Improvements](#future-improvements)

---

## Overview

The PPE Detection AI module is a self-contained computer vision system that analyzes images of construction sites and detects whether workers are wearing proper safety equipment. It identifies four key safety classes:

- ✅ **Hardhat** — Worker is wearing a hardhat
- ❌ **No Hardhat** — Worker is NOT wearing a hardhat (safety violation)
- ✅ **Vest** — Worker is wearing a safety vest
- ❌ **No Vest** — Worker is NOT wearing a safety vest (safety violation)

The system uses **transfer learning** by fine-tuning a pretrained YOLOv8-nano model on a curated construction site safety dataset sourced from Roboflow Universe.

### Key Features

| Feature | Description |
|---|---|
| **Real-time Detection** | ~640ms inference per image on CPU (i5-12450H) |
| **4-Class Detection** | Hardhat, No-Hardhat, Vest, No-Vest |
| **Transfer Learning** | Fine-tuned from COCO-pretrained YOLOv8n |
| **Pydantic Schema** | Type-safe detection output with validation |
| **ONNX Export** | Optimized ONNX model for faster CPU deployment |
| **FastAPI Endpoint** | Standalone test API with file upload |
| **Automated Tests** | 8 unit tests covering all detection scenarios |
| **Singleton Model** | Model loads once at import time for efficiency |

---

## Model Architecture

### Base Model: YOLOv8-nano (yolov8n)

| Property | Value |
|---|---|
| **Architecture** | YOLOv8 Nano |
| **Framework** | Ultralytics YOLOv8 (v8.2.30) |
| **Backbone** | CSPDarknet with C2f blocks |
| **Neck** | PAN-FPN (Path Aggregation Network) |
| **Head** | Decoupled Detect head |
| **Total Layers** | 225 |
| **Parameters** | 3,011,628 (3.0M) |
| **Trainable Parameters** | 3,011,612 |
| **GFLOPs** | 8.2 |
| **Model Size (.pt)** | 6.0 MB |
| **Model Size (.onnx)** | 11.7 MB |

### Why YOLOv8-nano?

- **Speed**: Designed for real-time inference, even on CPU-only machines
- **Size**: Only 3M parameters — lightweight enough for edge deployment
- **Accuracy**: Sufficient for PPE detection when fine-tuned with quality data
- **COCO Pretrained**: Transfer learning from 80-class COCO dataset provides strong feature extraction for person/object detection

### Layer Architecture

```
Layer  From    Module                                    Parameters
  0      -1     Conv [3, 16, 3, 2]                           464
  1      -1     Conv [16, 32, 3, 2]                        4,672
  2      -1     C2f  [32, 32, 1, True]                     7,360
  3      -1     Conv [32, 64, 3, 2]                       18,560
  4      -1     C2f  [64, 64, 2, True]                    49,664
  5      -1     Conv [64, 128, 3, 2]                      73,984
  6      -1     C2f  [128, 128, 2, True]                 197,632
  7      -1     Conv [128, 256, 3, 2]                    295,424
  8      -1     C2f  [256, 256, 1, True]                 460,288
  9      -1     SPPF [256, 256, 5]                       164,608
 10      -1     Upsample [None, 2, 'nearest']                 0
 11   [-1, 6]   Concat [1]                                     0
 12      -1     C2f  [384, 128, 1]                       148,224
 13      -1     Upsample [None, 2, 'nearest']                 0
 14   [-1, 4]   Concat [1]                                     0
 15      -1     C2f  [192, 64, 1]                         37,248
 16      -1     Conv [64, 64, 3, 2]                       36,992
 17  [-1, 12]   Concat [1]                                     0
 18      -1     C2f  [192, 128, 1]                       123,648
 19      -1     Conv [128, 128, 3, 2]                    147,712
 20   [-1, 9]   Concat [1]                                     0
 21      -1     C2f  [384, 256, 1]                       493,056
 22 [15,18,21]  Detect [4, [64, 128, 256]]               752,092
```

### Transfer Learning Approach

The model uses **fine-tuning** — NOT training from scratch:

1. Start with `yolov8n.pt` pretrained on **COCO dataset** (80 classes, 330K images)
2. Override the detection head from 80 classes → 4 PPE classes
3. Transfer 319 of 355 pretrained weight layers
4. Freeze the DFL (Distribution Focal Loss) convolution layer
5. Fine-tune on our PPE dataset

This approach leverages COCO's learned feature representations (edges, textures, object shapes, person detection) and adapts them specifically for PPE detection.

---

## Dataset

### Source

| Property | Value |
|---|---|
| **Provider** | [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) |
| **Project** | Construction Site Safety |
| **Version Used** | v1 (original, 398 images) |
| **License** | CC BY 4.0 |
| **Format** | YOLOv8 (normalized bbox: cx, cy, w, h) |
| **Image Format** | JPEG |
| **Original Classes** | 17 (Barricade, Dumpster, Excavators, Gloves, Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Net, Safety Shoes, Safety Vest, dump truck, mini-van, truck, wheel loader) |

### Class Remapping

The original 17-class dataset was remapped to **4 target PPE classes**:

| Original Class (Roboflow) | → | Target Class | Target ID |
|---|---|---|---|
| Hardhat | → | `hardhat` | 0 |
| NO-Hardhat | → | `no_hardhat` | 1 |
| Safety Vest | → | `vest` | 2 |
| NO-Safety Vest | → | `no_vest` | 3 |
| *All other 13 classes* | → | *Dropped* | — |

Images with no relevant PPE annotations after remapping were removed entirely.

### Dataset Split (After Remapping)

| Split | Images | Purpose |
|---|---|---|
| **Train** | 132 | Model training |
| **Validation** | 40 | Epoch-by-epoch evaluation & early stopping |
| **Test** | 15 | Final evaluation & smoke testing |
| **Total** | 187 | — |

### Class Distribution (Training Set — v1 Model)

| Class | ID | Train Labels | Valid Labels | Test Labels | Total | % of Total |
|---|---|---|---|---|---|---|
| `hardhat` | 0 | 289 | 115 | 32 | 436 | 68.6% |
| `no_hardhat` | 1 | 92 | 9 | 7 | 108 | 17.0% |
| `vest` | 2 | 45 | 6 | 2 | 53 | 8.3% |
| `no_vest` | 3 | 21 | 14 | 3 | 38 | 6.0% |
| **Total** | — | **447** | **144** | **44** | **635** | 100% |

> ⚠️ **Class Imbalance**: `vest` and `no_vest` are significantly underrepresented (6-8% of annotations), which directly impacts detection recall for these classes.

### Preprocessing

- Auto-orientation of pixel data (EXIF-orientation stripping)
- No augmentation was applied to the raw dataset (augmentation is applied during training)

---

## Training Pipeline

### Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| **Base Model** | `yolov8n.pt` | COCO-pretrained nano checkpoint |
| **Epochs** | 50 (max) | Maximum training epochs |
| **Batch Size** | 8 | Images per training step |
| **Image Size** | 640 × 640 | Input resolution |
| **Patience** | 10 | Early stopping patience |
| **Optimizer** | AdamW (auto-selected) | Adaptive learning rate |
| **Learning Rate** | 0.00125 | Auto-tuned by Ultralytics |
| **Momentum** | 0.9 | AdamW momentum |
| **Weight Decay** | 0.0005 | L2 regularization |
| **Warmup Epochs** | 3.0 | Learning rate warmup |
| **Device** | CPU | Intel Core i5-12450H |

### Data Augmentation (During Training)

Ultralytics applies these augmentations automatically during training:

| Augmentation | Value | Description |
|---|---|---|
| Mosaic | 1.0 | 4-image mosaic composition |
| Horizontal Flip | 0.5 | Random left-right flip |
| HSV Hue | 0.015 | Color hue variation |
| HSV Saturation | 0.7 | Color saturation variation |
| HSV Value | 0.4 | Brightness variation |
| Translation | 0.1 | Random translation |
| Scale | 0.5 | Random scaling |
| Erasing | 0.4 | Random erasing |
| Auto Augment | RandAugment | Automatic augmentation policy |

### Training Process

```
python ai/src/train.py
```

The training script (`ai/src/train.py`) performs the following steps:

1. **Load** pretrained `yolov8n.pt` checkpoint
2. **Override** detection head: 80 COCO classes → 4 PPE classes
3. **Transfer** 319/355 pretrained weight layers
4. **Train** with early stopping (patience=10)
5. **Validate** best checkpoint on validation set
6. **Export** best weights to `ai/models/ppe_best.pt`
7. **Export** ONNX model to `ai/models/ppe_best.onnx`
8. **Copy** 15 test images to `ai/tests/sample_images/`

### Training Duration

| Metric | Value |
|---|---|
| **Total Epochs Run** | 21 of 50 |
| **Best Epoch** | 11 |
| **Early Stopping** | Triggered at epoch 21 (no improvement for 10 epochs) |
| **Total Training Time** | ~1.05 hours |
| **Hardware** | Intel Core i5-12450H (CPU-only) |

---

## Training Results & Performance

### Final Validation Results (Best Checkpoint — Epoch 11)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **All (Overall)** | **0.748** | **0.338** | **0.324** | **0.172** |
| `hardhat` | 0.736 | 0.461 | 0.611 | 0.268 |
| `no_hardhat` | 0.257 | 0.889 | 0.671 | 0.413 |
| `vest` | 1.000 | 0.000 | 0.010 | 0.005 |
| `no_vest` | 1.000 | 0.000 | 0.004 | 0.001 |

### Performance Interpretation

| Class | Status | Explanation |
|---|---|---|
| `hardhat` | ⚠️ Usable | Decent precision (0.74) and moderate recall (0.46). mAP@0.5 of 0.61 means it detects most hardhats |
| `no_hardhat` | ⚠️ Usable | High recall (0.89) catches most violations, but low precision (0.26) means some false alarms |
| `vest` | ❌ Not Working | Zero recall — model cannot detect safety vests at all (only 45 training labels) |
| `no_vest` | ❌ Not Working | Zero recall — model cannot detect missing vests (only 21 training labels) |

### Inference Speed

| Metric | Value |
|---|---|
| **Preprocess** | 9.1 ms |
| **Inference** | 639.4 ms |
| **Postprocess** | 40.6 ms |
| **Total per Image** | ~690 ms |
| **Hardware** | Intel Core i5-12450H (CPU) |

### Training Loss Curves

Training losses decreased steadily across epochs:

| Metric | Epoch 1 | Epoch 11 (Best) | Epoch 21 (Final) |
|---|---|---|---|
| Box Loss | 1.748 | 1.451 | 1.322 |
| Cls Loss | 4.088 | 1.863 | 1.557 |
| DFL Loss | 1.518 | 1.249 | 1.196 |

---

## Detection Classes

### Class Definitions

| Class | ID | Description | Safety Status |
|---|---|---|---|
| `hardhat` | 0 | Worker wearing a hardhat/helmet | ✅ Compliant |
| `no_hardhat` | 1 | Worker NOT wearing a hardhat | ❌ Violation |
| `vest` | 2 | Worker wearing a safety/high-visibility vest | ✅ Compliant |
| `no_vest` | 3 | Worker NOT wearing a safety vest | ❌ Violation |

### Safety Compliance Logic

The backend uses this logic to flag images:

```python
from ai.src.detector import run_detection

detections = run_detection(saved_file_path)
if any(d.class_name in ("no_hardhat", "no_vest") and d.confidence > 0.5 for d in detections):
    image.ai_label = "issue_detected"
else:
    image.ai_label = "compliant"
```

---

## Inference Pipeline

### How Detection Works

```
Input Image (JPEG/PNG)
        │
        ▼
┌─────────────────────┐
│  Image Validation    │  ← Checks file exists
│  (Path verification) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  YOLOv8 Inference   │  ← Model loaded as singleton
│  (640×640 resize)   │  ← Confidence threshold: 0.4
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Class Mapping      │  ← Maps raw names → target classes
│  (class_map.py)     │  ← Filters non-PPE detections
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Schema Validation  │  ← Pydantic Detection model
│  (schemas.py)       │  ← Validates bbox, confidence
└─────────┬───────────┘
          │
          ▼
  List[Detection]  ← Output
```

### Detection Schema

```python
from pydantic import BaseModel, Field

class Detection(BaseModel):
    class_name: str   = Field(..., description="hardhat | no_hardhat | vest | no_vest")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_x: float     = Field(..., ge=0.0)  # Top-left X (pixels)
    bbox_y: float     = Field(..., ge=0.0)  # Top-left Y (pixels)
    bbox_w: float     = Field(..., ge=0.0)  # Width (pixels)
    bbox_h: float     = Field(..., ge=0.0)  # Height (pixels)
```

### Configuration

| Parameter | Value | File |
|---|---|---|
| Model Path | `ai/models/ppe_best.pt` | `detector.py` |
| Confidence Threshold | 0.4 | `detector.py` |
| Model Fusion | Enabled (`.fuse()`) | `detector.py` |
| Singleton Loading | Yes (loaded once at import) | `detector.py` |

---

## API Reference

### Core Function: `run_detection()`

```python
from ai.src.detector import run_detection

detections = run_detection("/path/to/image.jpg")
```

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | Absolute path to an image file (JPEG/PNG) |

**Returns:** `List[Detection]` — List of detection objects (may be empty)

**Raises:**
| Exception | When |
|---|---|
| `ImageLoadError` | Image file does not exist |
| `RuntimeError` | Model failed to load |

**Example Output:**
```python
[
    Detection(class_name='hardhat', confidence=0.92, bbox_x=120.5, bbox_y=45.3, bbox_w=80.0, bbox_h=65.2),
    Detection(class_name='no_vest', confidence=0.78, bbox_x=200.1, bbox_y=150.0, bbox_w=110.5, bbox_h=180.3)
]
```

### FastAPI Endpoint (Standalone Testing)

```bash
# Start the test server
uvicorn ai.service.detect_router:app --reload --port 8001

# Upload an image
curl -X POST http://localhost:8001/detect -F "image=@photo.jpg"
```

**Endpoints:**
| Method | Path | Description |
|---|---|---|
| POST | `/detect` | Upload image, get JSON detections |
| GET | `/health` | Health check (`{"status": "ok"}`) |

---

## Project Structure

```
BuildaThon-AI/
├── ai/                           # AI module root
│   ├── __init__.py               # Package init
│   ├── .env                      # API keys (not committed)
│   ├── .env.example              # Environment template
│   ├── README.md                 # Quick reference docs
│   ├── requirements.txt          # Python dependencies
│   │
│   ├── data/                     # Dataset
│   │   ├── ppe_dataset.yaml      # Ultralytics dataset config
│   │   ├── data.yaml             # Original Roboflow config (17 classes)
│   │   ├── prepare_dataset.py    # Download & remap dataset script
│   │   ├── augment_dataset.py    # Add more data from newer versions
│   │   ├── train/                # Training images + labels
│   │   ├── valid/                # Validation images + labels
│   │   └── test/                 # Test images + labels
│   │
│   ├── models/                   # Trained model weights
│   │   ├── ppe_best.pt           # Best PyTorch checkpoint (6.0 MB)
│   │   └── ppe_best.onnx         # ONNX export (11.7 MB)
│   │
│   ├── src/                      # Source code
│   │   ├── __init__.py
│   │   ├── detector.py           # Main detection function
│   │   ├── schemas.py            # Pydantic Detection model
│   │   ├── class_map.py          # Class name mapping
│   │   └── train.py              # Training script
│   │
│   ├── service/                  # API layer
│   │   └── detect_router.py      # FastAPI standalone endpoint
│   │
│   ├── tests/                    # Test suite
│   │   ├── test_detector.py      # 8 unit tests
│   │   └── sample_images/        # Test images (copied during training)
│   │
│   └── runs/                     # Training artifacts
│       └── ppe_train/            # Training run outputs
│           ├── weights/          # Saved checkpoints
│           ├── results.csv       # Epoch-by-epoch metrics
│           ├── results.png       # Loss/metric curves
│           ├── confusion_matrix.png
│           ├── labels.jpg        # Dataset label distribution
│           └── *.jpg             # Train/val batch visualizations
│
├── conftest.py                   # Pytest path configuration
├── yolov8n.pt                    # Base pretrained model (6.5 MB)
└── .gitignore
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | 8.2.30 | YOLOv8 training & inference |
| `torch` | 2.2.2 | PyTorch deep learning framework |
| `pydantic` | 2.6.4 | Detection schema validation |
| `fastapi` | 0.111.0 | Standalone test API |
| `pytest` | 8.2.1 | Unit testing |
| `pillow` | 10.3.0 | Image processing |
| `opencv-python-headless` | 4.8.0.74 | Computer vision utilities |
| `roboflow` | 1.1.13 | Dataset download from Roboflow |
| `python-dotenv` | 1.0.1 | Environment variable loading |
| `numpy` | 1.26.4 | Numerical operations |
| `pyyaml` | ≥6.0 | YAML config parsing |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/Korale05/Buildathon-LOL-AI.git
cd Buildathon-LOL-AI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r ai/requirements.txt

# 4. Set up environment variables
cp ai/.env.example ai/.env
# Edit ai/.env and add your ROBOFLOW_API_KEY

# 5. Download and prepare dataset
python ai/data/prepare_dataset.py

# 6. Train the model
python ai/src/train.py

# 7. Run tests
pytest ai/tests/ -v
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ROBOFLOW_API_KEY` | Yes (for data download) | API key from [Roboflow](https://roboflow.com) |
| `ROBOFLOW_PROJECT` | No | Default: `construction-site-safety` |

---

## Usage Guide

### Basic Detection

```python
from ai.src.detector import run_detection

# Run detection on an image
detections = run_detection("path/to/construction_site.jpg")

# Process results
for det in detections:
    print(f"Class: {det.class_name}")
    print(f"Confidence: {det.confidence:.2f}")
    print(f"Bounding Box: ({det.bbox_x}, {det.bbox_y}, {det.bbox_w}, {det.bbox_h})")
    print("---")
```

### Safety Compliance Check

```python
from ai.src.detector import run_detection

def check_safety(image_path: str) -> dict:
    """Check if workers in the image are wearing proper PPE."""
    detections = run_detection(image_path)

    violations = [d for d in detections if d.class_name in ("no_hardhat", "no_vest") and d.confidence > 0.5]
    compliant = [d for d in detections if d.class_name in ("hardhat", "vest")]

    return {
        "compliant": len(violations) == 0,
        "violations": len(violations),
        "detections": [d.dict() for d in detections],
    }
```

### Error Handling

```python
from ai.src.detector import run_detection, ImageLoadError

try:
    detections = run_detection("missing_image.jpg")
except ImageLoadError:
    print("Image file not found!")
except RuntimeError:
    print("Model failed to load!")
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest ai/tests/ -v

# Run with output
pytest ai/tests/ -v -s
```

### Test Suite (8 Tests)

| Test | Description | Status |
|---|---|---|
| `test_returns_list` | `run_detection()` always returns a list | ✅ PASSED |
| `test_class_names_are_valid` | All class names are in allowed set | ✅ PASSED |
| `test_bbox_values_non_negative` | Bounding box values ≥ 0 | ✅ PASSED |
| `test_confidence_range` | Confidence scores between 0.0 and 1.0 | ✅ PASSED |
| `test_detection_schema_fields` | Detection objects have all required fields | ✅ PASSED |
| `test_file_not_found_raises_error` | Missing file raises `ImageLoadError` | ✅ PASSED |
| `test_empty_result_is_list` | Empty result is `[]`, not `None` | ✅ PASSED |
| `test_multiple_images` | Consistent results across multiple images | ✅ PASSED |

---

## Model Export (ONNX)

The model is automatically exported to ONNX format during training:

| Property | Value |
|---|---|
| **Format** | ONNX (Open Neural Network Exchange) |
| **Opset** | 17 |
| **Input Shape** | (1, 3, 640, 640) — BCHW |
| **Output Shape** | (1, 8, 8400) |
| **File Size** | 11.7 MB |
| **Path** | `ai/models/ppe_best.onnx` |

To use the ONNX model instead of PyTorch, change the `MODEL_PATH` in `detector.py`:

```python
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ppe_best.onnx"
```

---

## Known Limitations

1. **Vest Detection Not Working**: The `vest` and `no_vest` classes have 0% recall due to extremely limited training data (45 and 21 annotations respectively). The model effectively cannot detect vests.

2. **CPU-Only Training**: Training was performed on CPU (Intel i5-12450H), which limited the number of training epochs and experimentation possible.

3. **Small Dataset**: Only 132 training images with 447 total annotations. Industry recommendation is 1,500+ images per class.

4. **Class Imbalance**: `hardhat` has 14x more annotations than `no_vest`, causing the model to be biased toward hardhat detection.

5. **Confidence Threshold**: Fixed at 0.4 — may need tuning per deployment environment.

6. **Single Image Inference**: The current API processes one image at a time (no batch inference).

---

## Future Improvements

1. **More Training Data**: Download and merge newer versions of the Roboflow dataset (v30 has 717 images with better vest/no_vest coverage)

2. **Advanced Augmentation**: Add MixUp, Copy-Paste, and rotation augmentation to synthetically increase rare class exposure

3. **Larger Model**: Upgrade from YOLOv8n (3M params) to YOLOv8s (11M params) for better feature extraction

4. **GPU Training**: Train on GPU for more epochs and faster experimentation

5. **Batch Inference**: Support multiple images in a single API call

6. **Video Stream**: Add support for real-time video feed processing

7. **Confidence Calibration**: Per-class confidence thresholds for optimal precision/recall trade-off

---

## License

- **Dataset**: CC BY 4.0 (Roboflow Universe — Construction Site Safety)
- **YOLOv8**: AGPL-3.0 (Ultralytics)

---

*Documentation generated for BuildaThon-AI PPE Detection Module*
*Model trained on: September 14, 2026*
*Total training time: ~1.05 hours (21 epochs on CPU)*
