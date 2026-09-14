# PPE Detection AI Component

## Overview
Self-contained PPE detection module for the Construction Site Intelligence Platform.

**Import**: `from ai.src.detector import run_detection`

## Quick Start

```python
from ai.src.detector import run_detection

detections = run_detection("/uploads/photo123.jpg")
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f} at ({det.bbox_x}, {det.bbox_y}, {det.bbox_w}, {det.bbox_h})")
```

## Detection Schema

```python
class Detection(BaseModel):
    class_name: str   # "hardhat" | "no_hardhat" | "vest" | "no_vest"
    confidence: float # 0.0 - 1.0
    bbox_x: float     # top-left x (absolute pixels)
    bbox_y: float     # top-left y (absolute pixels)
    bbox_w: float     # width (pixels)
    bbox_h: float     # height (pixels)
```

## Backend Integration

The backend flags images using this logic:
```python
detections = run_detection(saved_file_path)
if any(d.class_name in ("no_hardhat", "no_vest") and d.confidence > 0.5 for d in detections):
    image.ai_label = "issue_detected"
else:
    image.ai_label = "compliant"
```

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv8 nano (yolov8n) |
| Dataset | Construction Site Safety (Roboflow), remapped to 4 classes |
| Training | 50 epochs, batch 8, patience 10, imgsz 640 |
| Classes | hardhat, no_hardhat, vest, no_vest |
| Inference device | CPU (GPU optional) |
| Typical latency | ~100-200ms per image on CPU |

## Dataset Class Balance

| Class | Train | Valid | Test | Total |
|---|---|---|---|---|
| hardhat | 289 | 115 | 32 | 436 |
| no_hardhat | 92 | 9 | 7 | 108 |
| vest | 45 | 6 | 2 | 53 |
| no_vest | 21 | 14 | 3 | 38 |

> **Note**: vest and no_vest are underrepresented. Recall on these classes may be lower.

## ONNX Model

An ONNX export (`ai/models/ppe_best.onnx`) is generated during training.
To switch to ONNX for faster CPU inference, change `MODEL_PATH` in `detector.py`:
```python
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ppe_best.onnx"
```

## Re-training

```bash
python ai/src/train.py
```

This overwrites `ai/models/ppe_best.pt` and `ai/models/ppe_best.onnx`.

## Standalone Testing (for development only)

```bash
uvicorn ai.service.detect_router:app --reload --port 8001
# POST /detect with multipart image upload
curl -X POST http://localhost:8001/detect -F "image=@path/to/image.jpg"
```

## Running Tests

```bash
pytest ai/tests/ -v
```

## Hardware Requirements

- **CPU**: Works out of the box, ~100-200ms per 640x640 image
- **GPU**: Optional, set `DEVICE = "0"` in `train.py` for faster training
- **RAM**: ~2GB for inference, ~4GB for training
