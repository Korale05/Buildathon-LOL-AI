from pydantic import BaseModel, Field

class Detection(BaseModel):
    """Detection schema matching the backend's `detections` table.

    Attributes
    ----------
    class_name: str
        One of "hardhat", "no_hardhat", "vest", "no_vest".
    confidence: float
        Confidence score from the model (0.0 – 1.0).
    bbox_x: float
        Top‑left x coordinate in absolute pixel units.
    bbox_y: float
        Top‑left y coordinate in absolute pixel units.
    bbox_w: float
        Width of the bounding box in pixels.
    bbox_h: float
        Height of the bounding box in pixels.
    """

    class_name: str = Field(..., description="hardhat | no_hardhat | vest | no_vest")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_x: float = Field(..., ge=0.0)
    bbox_y: float = Field(..., ge=0.0)
    bbox_w: float = Field(..., ge=0.0)
    bbox_h: float = Field(..., ge=0.0)
