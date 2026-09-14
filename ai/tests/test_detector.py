"""
Unit tests for the PPE detection module.

Run with:
    pytest ai/tests/test_detector.py -v
"""

import os
import pytest
from pathlib import Path

# We need to set up the path so imports work
AI_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = AI_ROOT / "tests" / "sample_images"
MODEL_PATH = AI_ROOT / "models" / "ppe_best.pt"

ALLOWED_CLASSES = {"hardhat", "no_hardhat", "vest", "no_vest"}


def _get_sample_images():
    """Get list of sample images for testing."""
    if not SAMPLE_DIR.exists():
        return []
    return list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png"))


# Skip all tests if model is not available
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Model not found at {MODEL_PATH}. Run 'python ai/src/train.py' first."
)


class TestRunDetection:
    """Tests for the run_detection function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import run_detection (loads model once)."""
        from ai.src.detector import run_detection, ImageLoadError
        self.run_detection = run_detection
        self.ImageLoadError = ImageLoadError

    def test_returns_list(self):
        """run_detection should always return a list."""
        images = _get_sample_images()
        if not images:
            pytest.skip("No sample images available")
        result = self.run_detection(str(images[0]))
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_class_names_are_valid(self):
        """Every detection's class_name must be one of the 4 allowed values."""
        images = _get_sample_images()
        if not images:
            pytest.skip("No sample images available")
        for img in images[:5]:
            detections = self.run_detection(str(img))
            for det in detections:
                assert det.class_name in ALLOWED_CLASSES, (
                    f"Invalid class_name '{det.class_name}' for image {img.name}. "
                    f"Allowed: {ALLOWED_CLASSES}"
                )

    def test_bbox_values_non_negative(self):
        """All bbox values should be non-negative floats."""
        images = _get_sample_images()
        if not images:
            pytest.skip("No sample images available")
        for img in images[:5]:
            detections = self.run_detection(str(img))
            for det in detections:
                assert det.bbox_x >= 0, f"bbox_x is negative: {det.bbox_x}"
                assert det.bbox_y >= 0, f"bbox_y is negative: {det.bbox_y}"
                assert det.bbox_w >= 0, f"bbox_w is negative: {det.bbox_w}"
                assert det.bbox_h >= 0, f"bbox_h is negative: {det.bbox_h}"

    def test_confidence_range(self):
        """Confidence scores should be between 0 and 1."""
        images = _get_sample_images()
        if not images:
            pytest.skip("No sample images available")
        for img in images[:5]:
            detections = self.run_detection(str(img))
            for det in detections:
                assert 0.0 <= det.confidence <= 1.0, (
                    f"Confidence {det.confidence} out of range [0, 1]"
                )

    def test_detection_schema_fields(self):
        """Each Detection should have all required fields."""
        images = _get_sample_images()
        if not images:
            pytest.skip("No sample images available")
        result = self.run_detection(str(images[0]))
        if result:
            det = result[0]
            assert hasattr(det, "class_name")
            assert hasattr(det, "confidence")
            assert hasattr(det, "bbox_x")
            assert hasattr(det, "bbox_y")
            assert hasattr(det, "bbox_w")
            assert hasattr(det, "bbox_h")

    def test_file_not_found_raises_error(self):
        """Passing a non-existent file should raise ImageLoadError."""
        with pytest.raises(self.ImageLoadError):
            self.run_detection("nonexistent_image_12345.jpg")

    def test_empty_result_is_list(self):
        """Even with no detections, result should be a list (not None)."""
        images = _get_sample_images()
        if not images:
            pytest.skip("No sample images available")
        # We can't guarantee no detections, but we verify the type
        result = self.run_detection(str(images[0]))
        assert result is not None
        assert isinstance(result, list)

    def test_multiple_images(self):
        """Run detection on multiple images to check consistency."""
        images = _get_sample_images()
        if len(images) < 2:
            pytest.skip("Need at least 2 sample images")
        for img in images[:5]:
            result = self.run_detection(str(img))
            assert isinstance(result, list)
            for det in result:
                assert det.class_name in ALLOWED_CLASSES
                assert isinstance(det.confidence, float)
                assert isinstance(det.bbox_x, float)
                assert isinstance(det.bbox_y, float)
                assert isinstance(det.bbox_w, float)
                assert isinstance(det.bbox_h, float)
