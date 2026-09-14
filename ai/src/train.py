"""
Fine-tune YOLOv8n on the remapped 4-class PPE dataset.

Usage:
    python ai/src/train.py

Outputs:
    ai/models/ppe_best.pt   - best checkpoint (by val mAP)
    ai/models/ppe_best.onnx - ONNX export for faster CPU inference
"""

import shutil
import logging
from pathlib import Path

from ultralytics import YOLO

# ── Paths ───────────────────────────────────────────────────────────────
AI_ROOT = Path(__file__).resolve().parents[1]           # ai/
DATASET_YAML = AI_ROOT / "data" / "ppe_dataset.yaml"
MODELS_DIR = AI_ROOT / "models"
BEST_PT = MODELS_DIR / "ppe_best.pt"
BEST_ONNX = MODELS_DIR / "ppe_best.onnx"

# ── Training hyperparameters (CPU-friendly) ─────────────────────────────
BASE_MODEL = "yolov8n.pt"          # pretrained COCO nano checkpoint
IMGSZ = 640
EPOCHS = 100                       # more epochs for better convergence
BATCH = 8                          # safe for CPU / low-RAM machines
PATIENCE = 20                      # early stopping (more patient)
DEVICE = "cpu"                     # change to "0" for GPU

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train():
    """Run fine-tuning and save results."""

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATASET_YAML}\n"
            "Run 'python ai/data/prepare_dataset.py' first."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load base model ─────────────────────────────────────────────────
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # ── Train ───────────────────────────────────────────────────────────
    logger.info("Starting training...")
    logger.info(f"  Dataset:   {DATASET_YAML}")
    logger.info(f"  Epochs:    {EPOCHS}")
    logger.info(f"  Batch:     {BATCH}")
    logger.info(f"  Image sz:  {IMGSZ}")
    logger.info(f"  Patience:  {PATIENCE}")
    logger.info(f"  Device:    {DEVICE}")

    results = model.train(
        data=str(DATASET_YAML),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        project=str(AI_ROOT / "runs"),
        name="ppe_train",
        exist_ok=True,
        # ── Data augmentation (boosted for class imbalance) ─────────────
        mosaic=1.0,          # Mosaic augmentation (ON by default)
        mixup=0.3,           # MixUp: blends images — helps rare classes
        copy_paste=0.3,      # Copy-paste: copies objects across images
        degrees=15.0,        # Random rotation ±15°
        scale=0.7,           # Aggressive scaling (default 0.5)
        flipud=0.3,          # Vertical flip (default 0.0)
        fliplr=0.5,          # Horizontal flip (default)
        cos_lr=True,         # Cosine learning rate — smoother convergence
        verbose=True,
    )

    # ── Copy best weights ───────────────────────────────────────────────
    train_dir = Path(model.trainer.save_dir)
    best_src = train_dir / "weights" / "best.pt"

    if best_src.exists():
        shutil.copy2(str(best_src), str(BEST_PT))
        logger.info(f"Best weights saved to: {BEST_PT}")
    else:
        # Fall back to last.pt
        last_src = train_dir / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy2(str(last_src), str(BEST_PT))
            logger.info(f"Last weights saved to: {BEST_PT} (best.pt not found)")
        else:
            logger.error("No weights found after training!")
            return

    # ── Validate ────────────────────────────────────────────────────────
    logger.info("Running validation on best model...")
    best_model = YOLO(str(BEST_PT))
    val_results = best_model.val(data=str(DATASET_YAML), device=DEVICE)

    # Print per-class metrics
    print("\n" + "=" * 65)
    print("VALIDATION RESULTS (best checkpoint)")
    print("=" * 65)
    class_names = ["hardhat", "no_hardhat", "vest", "no_vest"]

    # Access metrics
    metrics = val_results
    print(f"\n  Overall mAP@0.5:    {metrics.box.map50:.4f}")
    print(f"  Overall mAP@0.5:95: {metrics.box.map:.4f}")

    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10}")
    print("  " + "-" * 50)

    for i, cls_name in enumerate(class_names):
        if i < len(metrics.box.p):
            p = metrics.box.p[i]
            r = metrics.box.r[i]
            m = metrics.box.maps[i] if hasattr(metrics.box, 'maps') else 0.0
            print(f"  {cls_name:<15} {p:>10.4f} {r:>10.4f} {m:>10.4f}")

            # Flag low recall on safety-critical classes
            if cls_name in ("no_hardhat", "no_vest") and r < 0.70:
                print(f"  [!] WARNING: {cls_name} recall ({r:.2f}) is below 0.70!")
                print(f"      Consider retraining with class-weighted loss or oversampling.")

    # ── Export ONNX ─────────────────────────────────────────────────────
    logger.info("Exporting ONNX model...")
    try:
        best_model.export(format="onnx", imgsz=IMGSZ)
        # Ultralytics saves ONNX next to the .pt file; move it to our models dir
        exported_onnx = BEST_PT.with_suffix(".onnx")
        if exported_onnx.exists():
            logger.info(f"ONNX model saved to: {exported_onnx}")
        else:
            logger.warning("ONNX export completed but file not found at expected path.")
    except Exception as e:
        logger.warning(f"ONNX export failed (non-critical): {e}")
        logger.warning("You can export later with: model.export(format='onnx')")

    # ── Copy sample test images for smoke testing ───────────────────────
    test_images_dir = AI_ROOT / "data" / "test" / "images"
    sample_dir = AI_ROOT / "tests" / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if test_images_dir.exists():
        test_imgs = list(test_images_dir.glob("*.jpg"))[:15]  # up to 15 images
        for img in test_imgs:
            shutil.copy2(str(img), str(sample_dir / img.name))
        logger.info(f"Copied {len(test_imgs)} test images to {sample_dir}")
    else:
        logger.warning(f"No test images found at {test_images_dir}")

    print("\n" + "=" * 65)
    print("[OK] Training complete!")
    print(f"  Best weights: {BEST_PT}")
    print(f"  ONNX model:   {BEST_ONNX}")
    print(f"  Training logs: {train_dir}")
    print(f"  Sample images: {sample_dir}")
    print("\n  Next steps:")
    print("    1. Review per-class recall above (especially no_hardhat, no_vest)")
    print("    2. Run tests:  pytest ai/tests/")
    print("    3. Try inference: python -c \"from ai.src.detector import run_detection; print(run_detection('path/to/image.jpg'))\"")
    print("=" * 65)


if __name__ == "__main__":
    train()
