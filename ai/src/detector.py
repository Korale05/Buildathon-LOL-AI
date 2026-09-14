import logging
import time
from pathlib import Path
from typing import List

from ultralytics import YOLO
from pydantic import ValidationError

from .schemas import Detection
from .class_map import TARGET_CLASSES, normalize_class

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Constants
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ppe_best.pt"
PRETRAINED_MODEL_URL = "https://public.roboflow.com/dataset/construction-site-safety/model"  # placeholder URL
CONF_THRESHOLD = 0.4


class ImageLoadError(FileNotFoundError):
    """Raised when the provided image path does not exist or cannot be opened."""

# Load model at import time (singleton)
def _load_model() -> YOLO:
    if MODEL_PATH.is_file():
        logger.info(f"Loading PPE model from {MODEL_PATH}")
        return YOLO(str(MODEL_PATH))
    else:
        # Fallback: attempt to download a pretrained checkpoint from Roboflow Universe
        try:
            from roboflow import Roboflow
            import os
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if not api_key:
                raise RuntimeError("ROBOFLOW_API_KEY not set; cannot download pretrained model")
            rf = Roboflow(api_key=api_key)
            # The exact workspace and model name may vary; using generic call
            project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
            # Assume the model is available as a YOLOv8 checkpoint named "ppe-detection"
            model = project.version(1).model("ppe-detection").download("yolov8")
            pretrained_path = Path(model.location) / "weights" / "best.pt"
            if pretrained_path.is_file():
                logger.info(f"Using pretrained checkpoint from Roboflow: {pretrained_path}")
                return YOLO(str(pretrained_path))
            else:
                raise FileNotFoundError("Downloaded pretrained model not found")
        except Exception as e:
            logger.error(f"Failed to download pretrained model: {e}")
            raise RuntimeError("No model available for inference")

_model = _load_model()
_model.fuse()
_model.conf = CONF_THRESHOLD


def _convert_xyxy_to_bbox(xyxy: List[float]) -> tuple[float, float, float, float]:
    """Convert YOLO xyxy (pixel) to (x, y, w, h).

    Parameters
    ----------
    xyxy: list of four floats [x1, y1, x2, y2]
        Absolute pixel coordinates of the box corners.

    Returns
    -------
    (bbox_x, bbox_y, bbox_w, bbox_h)
    """
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return float(x1), float(y1), float(w), float(h)


def run_detection(image_path: str) -> List[Detection]:
    """Run PPE detection on a single image.

    Parameters
    ----------
    image_path: str
        Path to an image file on disk.

    Returns
    -------
    List[Detection]
        A list of :class:`Detection` objects. The list may be empty if no
        relevant objects are found.

    Raises
    ------
    ImageLoadError
        If the image file does not exist or cannot be opened by the model.
    RuntimeError
        If model loading failed.
    ValidationError
        If the produced detections do not conform to the schema (should never happen).
    """
    img_file = Path(image_path)
    if not img_file.is_file():
        raise ImageLoadError(f"Image file not found: {image_path}")

    start = time.time()
    results = _model(str(img_file))  # YOLO inference
    inference_time = (time.time() - start) * 1000  # ms

    detections: List[Detection] = []
    if results and len(results) > 0:
        # YOLO returns a list with one element per image; we have only one image
        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls[0]) if box.cls is not None else None
            # Ultralytics provides class name via model.names mapping
            raw_name = _model.names.get(class_id, None) if class_id is not None else None
            # If the model's class name is already one of the target classes, use it directly.
            if raw_name in TARGET_CLASSES:
                target_name = raw_name
            else:
                # Otherwise attempt to map from a raw dataset label.
                target_name = normalize_class(raw_name) if raw_name else None
            if target_name is None or target_name not in TARGET_CLASSES:
                continue  # filter out unwanted classes
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            # Convert coordinates
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            bbox_x, bbox_y, bbox_w, bbox_h = _convert_xyxy_to_bbox(xyxy)
            det = Detection(
                class_name=target_name,
                confidence=conf,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
            )
            detections.append(det)

    logger.info(
        f"run_detection: {len(detections)} detections, inference_time={inference_time:.1f}ms for {image_path}"
    )
    return detections
