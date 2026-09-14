"""
Prepare the Construction Site Safety dataset for PPE detection.

This script:
  1. Downloads the dataset from Roboflow (requires ROBOFLOW_API_KEY in ai/.env)
  2. Remaps the raw 17-class labels to only 4 target classes:
       0: hardhat,  1: no_hardhat,  2: vest,  3: no_vest
  3. Drops all other class annotations, removes images with no relevant objects
  4. Writes ppe_dataset.yaml for Ultralytics training
  5. Prints a class-balance report

Safe to re-run: always downloads fresh and remaps from scratch.
"""

import os
import sys
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv


# ── Class mapping ───────────────────────────────────────────────────────
# Raw class names from Roboflow's data.yaml -> our target class IDs
#  "Hardhat"          -> 0 (hardhat)
#  "NO-Hardhat"       -> 1 (no_hardhat)
#  "Safety Vest"      -> 2 (vest)
#  "NO-Safety Vest"   -> 3 (no_vest)
# Everything else is dropped.

RAW_NAME_TO_TARGET_ID = {
    "Hardhat": 0,
    "NO-Hardhat": 1,
    "Safety Vest": 2,
    "NO-Safety Vest": 3,
}

TARGET_CLASSES = ["hardhat", "no_hardhat", "vest", "no_vest"]


def download_dataset(api_key: str) -> Path:
    """Download the Construction Site Safety dataset from Roboflow."""
    from roboflow import Roboflow

    project_slug = os.getenv("ROBOFLOW_PROJECT", "construction-site-safety")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project(project_slug)
    dataset = project.version(1).download("yolov8")
    return Path(dataset.location)


def build_raw_id_to_target_id(data_yaml_path: Path) -> dict:
    """Read the raw data.yaml and build raw_class_id -> target_class_id map."""
    with open(data_yaml_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    raw_names = raw_cfg.get("names", [])
    mapping = {}
    for raw_id, name in enumerate(raw_names):
        if name in RAW_NAME_TO_TARGET_ID:
            mapping[raw_id] = RAW_NAME_TO_TARGET_ID[name]
    print(f"  Mapping {len(mapping)} raw class IDs to target IDs:")
    for raw_id, target_id in sorted(mapping.items()):
        raw_name = raw_names[raw_id]
        print(f"    {raw_id} ({raw_name}) -> {target_id} ({TARGET_CLASSES[target_id]})")
    return mapping


def remap_labels(data_root: Path, mapping: dict):
    """Remap label files, keeping only the 4 target classes."""
    total_remapped = 0
    total_dropped_lines = 0
    total_removed_images = 0

    for split in ["train", "valid", "test"]:
        img_dir = data_root / split / "images"
        label_dir = data_root / split / "labels"

        if not label_dir.exists():
            print(f"  WARNING: {label_dir} does not exist, skipping.")
            continue

        label_files = list(label_dir.glob("*.txt"))
        print(f"  {split}: processing {len(label_files)} label files...")

        for label_path in label_files:
            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    raw_class_id = int(parts[0])
                    if raw_class_id in mapping:
                        new_id = mapping[raw_class_id]
                        # parts[1:] are the normalized bbox coords (cx, cy, w, h)
                        new_line = " ".join([str(new_id)] + parts[1:])
                        new_lines.append(new_line)
                        total_remapped += 1
                    else:
                        total_dropped_lines += 1

            # Overwrite label file with only the kept annotations
            with open(label_path, "w") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

            # If the label file is now empty, remove the image and the label
            if not new_lines:
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_path = img_dir / (label_path.stem + ext)
                    if img_path.exists():
                        img_path.unlink()
                        break
                label_path.unlink()
                total_removed_images += 1

    print(f"\n  Annotations kept (remapped): {total_remapped}")
    print(f"  Annotations dropped (other classes): {total_dropped_lines}")
    print(f"  Images removed (no relevant objects): {total_removed_images}")


def write_dataset_yaml(data_root: Path):
    """Write ppe_dataset.yaml for Ultralytics training."""
    yaml_path = data_root / "ppe_dataset.yaml"
    cfg = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 4,
        "names": TARGET_CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    print(f"\n  Wrote dataset config: {yaml_path}")


def print_class_balance(data_root: Path):
    """Print per-class annotation counts across all splits."""
    counts = {cls: 0 for cls in TARGET_CLASSES}
    split_counts = {}

    for split in ["train", "valid", "test"]:
        label_dir = data_root / split / "labels"
        split_counts[split] = {cls: 0 for cls in TARGET_CLASSES}
        if not label_dir.exists():
            continue
        for label_path in label_dir.glob("*.txt"):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if 0 <= class_id < len(TARGET_CLASSES):
                        cls_name = TARGET_CLASSES[class_id]
                        counts[cls_name] += 1
                        split_counts[split][cls_name] += 1

    print("\n" + "=" * 55)
    print("CLASS BALANCE REPORT")
    print("=" * 55)
    header = f"{'Class':<15} {'Train':>8} {'Valid':>8} {'Test':>8} {'Total':>8}"
    print(header)
    print("-" * 55)
    for cls in TARGET_CLASSES:
        row = (
            f"{cls:<15} "
            f"{split_counts.get('train', {}).get(cls, 0):>8} "
            f"{split_counts.get('valid', {}).get(cls, 0):>8} "
            f"{split_counts.get('test', {}).get(cls, 0):>8} "
            f"{counts[cls]:>8}"
        )
        print(row)
    print("-" * 55)
    total = sum(counts.values())
    print(f"{'TOTAL':<15} {' ':>8} {' ':>8} {' ':>8} {total:>8}")

    low = [cls for cls, cnt in counts.items() if cnt < 100]
    if low:
        print(f"\n  [!] WARNING: Classes with fewer than 100 instances: {', '.join(low)}")
        print("     These are safety-critical classes - consider oversampling or")
        print("     using class-weighted loss during training.")


def main():
    ai_root = Path(__file__).resolve().parent.parent  # ai/
    data_root = ai_root / "data"

    print("=" * 55)
    print("PPE Dataset Preparation")
    print("=" * 55)

    # Load .env for API key
    env_path = ai_root / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not set in ai/.env")
        sys.exit(1)

    # Step 1: Download dataset
    print("\n[1/4] Downloading dataset from Roboflow...")
    download_path = download_dataset(api_key)
    print(f"  Downloaded to: {download_path}")

    # Step 2: Copy to ai/data/ (preserving our scripts)
    print("\n[2/4] Setting up data directory...")
    # Remove old train/valid/test if they exist
    for split in ["train", "valid", "test"]:
        old = data_root / split
        if old.exists():
            shutil.rmtree(old)
    # Copy fresh splits from downloaded location
    for split in ["train", "valid", "test"]:
        src = download_path / split
        dst = data_root / split
        if src.exists():
            shutil.copytree(str(src), str(dst))
            print(f"  Copied {split}/")
        else:
            print(f"  WARNING: {src} not found in download")

    # Read the raw data.yaml from the download to build the class mapping
    raw_yaml = download_path / "data.yaml"
    if not raw_yaml.exists():
        # Try in our data dir
        raw_yaml = data_root / "data.yaml"
    print(f"\n  Reading class names from: {raw_yaml}")
    mapping = build_raw_id_to_target_id(raw_yaml)

    # Step 3: Remap labels
    print("\n[3/4] Remapping labels to 4 target classes...")
    remap_labels(data_root, mapping)

    # Step 4: Write yaml and report
    print("\n[4/4] Writing dataset config and computing balance...")
    write_dataset_yaml(data_root)
    print_class_balance(data_root)

    print("\n[OK] Dataset preparation complete!")
    print(f"   Dataset root: {data_root}")
    print(f"   Config file:  {data_root / 'ppe_dataset.yaml'}")
    print("\n   Next step: python ai/src/train.py")


if __name__ == "__main__":
    main()
