"""
Optional standalone FastAPI router for testing PPE detection independently.

THIS IS FOR LOCAL TESTING ONLY.
The real integration point is `from ai.src.detector import run_detection`.

Usage:
    uvicorn ai.service.detect_router:app --reload --port 8001

Test:
    curl -X POST http://localhost:8001/detect -F "image=@path/to/image.jpg"
"""

import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ai.src.detector import run_detection, ImageLoadError
from ai.src.schemas import Detection

app = FastAPI(
    title="PPE Detection - Standalone Test Service",
    description="For local testing only. Import run_detection() directly for production.",
    version="1.0.0",
)


@app.post("/detect", response_model=list[Detection])
async def detect_ppe(image: UploadFile = File(...)):
    """
    Upload an image and get PPE detection results.

    Returns a JSON list of Detection objects with:
    - class_name: hardhat | no_hardhat | vest | no_vest
    - confidence: float (0-1)
    - bbox_x, bbox_y: top-left corner in pixels
    - bbox_w, bbox_h: box dimensions in pixels
    """
    # Save uploaded file to a temp location
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        detections = run_detection(tmp_path)
        return detections
    except ImageLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": "ppe_best.pt"}
