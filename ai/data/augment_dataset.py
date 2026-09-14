"""
Augment PPE dataset by downloading a newer version from Roboflow.

This script:
  1. Downloads Construction Site Safety dataset v30 (717 images vs v1's 398)
  2. Remaps labels to same 4 classes: hardhat, no_hardhat, vest, no_vest
  3. Merges NEW images into existing train/valid/test splits (no duplicates)
  4. Deletes label cache so YOLO re-scans on next training
  5. Prints before/after class balance report

Usage:
    python ai/data/augment_dataset.py

After running:
    python ai/src/train.py
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict

import yaml
from dotenv import load_dotenv


# ── Same class mapping as prepare_dataset.py ────────────────────────────
RAW_NAME_TO_TARGET_ID = {
    "Hardhat": 0,
    "NO-Hardhat": 1,
    "Safety Vest": 2,
    "NO-Safety Vest": 3,
}
TARGET_CLASSES = ["hardhat", "no_hardhat", "vest", "no_vest"]

# Which Roboflow version to download (30 has 717 images)
ROBOFLOW_VERSION = 30


def count_classes(data_root: Path) -> dict:
    """Count per-class annotations across train/valid/test."""
    counts = defaultdict(lambda: defaultdict(int))
    for split in ["train", "valid", "test"]:
        label_dir = data_root / split / "labels"
        if not label_dir.exists():
            continue
        for label_path in label_dir.glob("*.txt"):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(TARGET_CLASSES):
                            counts[split][TARGET_CLASSES[class_id]] += 1
    return counts


def print_balance(title: str, counts: dict):
    """Pretty-print a class balance table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  {'Class':<15} {'Train':>8} {'Valid':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-' * 55}")
    for cls in TARGET_CLASSES:
        t = counts.get("train", {}).get(cls, 0)
        v = counts.get("valid", {}).get(cls, 0)
        te = counts.get("test", {}).get(cls, 0)
        total = t + v + te
        print(f"  {cls:<15} {t:>8} {v:>8} {te:>8} {total:>8}")
    grand = sum(
        counts.get(s, {}).get(c, 0)
        for s in ["train", "valid", "test"]
        for c in TARGET_CLASSES
    )
    print(f"  {'-' * 55}")
    print(f"  {'TOTAL':<15} {'':>8} {'':>8} {'':>8} {grand:>8}")


def download_v30(api_key: str, download_dir: Path) -> Path:
    """Download Construction Site Safety v30 from Roboflow."""
    from roboflow import Roboflow

    print(f"\n  Downloading v{ROBOFLOW_VERSION} from Roboflow...")
    print("  (This may take a few minutes)\n")

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace("roboflow-universe-projects")
    proj = ws.project("construction-site-safety")
    dataset = proj.version(ROBOFLOW_VERSION).download(
        "yolov8",
        location=str(download_dir),
    )
    return Path(dataset.location)


def build_mapping(data_yaml_path: Path) -> dict:
    """Build raw_class_id -> target_class_id mapping from data.yaml."""
    with open(data_yaml_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    raw_names = raw_cfg.get("names", [])

    # Handle both list and dict formats
    if isinstance(raw_names, dict):
        mapping = {}
        for raw_id, name in raw_names.items():
            if name in RAW_NAME_TO_TARGET_ID:
                mapping[int(raw_id)] = RAW_NAME_TO_TARGET_ID[name]
    else:
        mapping = {}
        for raw_id, name in enumerate(raw_names):
            if name in RAW_NAME_TO_TARGET_ID:
                mapping[raw_id] = RAW_NAME_TO_TARGET_ID[name]

    print(f"  Class mapping ({len(mapping)} classes):")
    for raw_id, target_id in sorted(mapping.items()):
        print(f"    raw {raw_id} -> {target_id} ({TARGET_CLASSES[target_id]})")
    return mapping


def merge_new_data(data_root: Path, new_data_root: Path, mapping: dict):
    """
    Remap labels in the new dataset and merge into existing data.
    Only adds images that don't already exist (by filename stem).
    """
    added = {"train": 0, "valid": 0, "test": 0}
    skipped = {"train": 0, "valid": 0, "test": 0}
    annotations_added = 0
    empty_removed = 0

    for split in ["train", "valid", "test"]:
        new_img_dir = new_data_root / split / "images"
        new_label_dir = new_data_root / split / "labels"

        if not new_img_dir.exists() or not new_label_dir.exists():
            print(f"  {split}: not found in new data, skipping")
            continue

        # Target directories
        target_img_dir = data_root / split / "images"
        target_label_dir = data_root / split / "labels"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        # Get existing image stems to avoid duplicates
        existing_stems = {p.stem for p in target_img_dir.glob("*.*")}

        new_labels = list(new_label_dir.glob("*.txt"))
        print(f"  {split}: found {len(new_labels)} label files in new data")

        for label_path in new_labels:
            stem = label_path.stem

            # Skip if already exists
            if stem in existing_stems:
                skipped[split] += 1
                continue

            # Remap labels
            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    raw_class_id = int(parts[0])
                    if raw_class_id in mapping:
                        new_id = mapping[raw_class_id]
                        new_line = " ".join([str(new_id)] + parts[1:])
                        new_lines.append(new_line)

            # Skip images with no relevant PPE annotations
            if not new_lines:
                empty_removed += 1
                continue

            # Find corresponding image
            img_found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = new_img_dir / (stem + ext)
                if img_path.exists():
                    # Copy image
                    shutil.copy2(str(img_path), str(target_img_dir / img_path.name))
                    # Write remapped label
                    target_label = target_label_dir / label_path.name
                    with open(target_label, "w") as f:
                        f.write("\n".join(new_lines) + "\n")
                    added[split] += 1
                    annotations_added += len(new_lines)
                    img_found = True
                    break

            if not img_found:
                # Image might have different extension
                pass

    print(f"\n  Results:")
    print(f"    Images added:   train={added['train']}, valid={added['valid']}, test={added['test']}")
    print(f"    Images skipped (duplicates): train={skipped['train']}, valid={skipped['valid']}, test={skipped['test']}")
    print(f"    Images removed (no PPE objects): {empty_removed}")
    print(f"    Total annotations added: {annotations_added}")

    return added


def clear_label_cache(data_root: Path):
    """Delete .cache files so YOLO re-scans the data."""
    for cache in data_root.rglob("*.cache"):
        cache.unlink()
        print(f"  Deleted cache: {cache}")
    # Also delete any .cache files in label dirs
    for split in ["train", "valid", "test"]:
        cache_file = data_root / split / "labels.cache"
        if cache_file.exists():
            cache_file.unlink()
            print(f"  Deleted cache: {cache_file}")


def main():
    ai_root = Path(__file__).resolve().parent.parent  # ai/
    data_root = ai_root / "data"

    print("=" * 60)
    print("  PPE Dataset Augmentation — Adding More Data")
    print("=" * 60)

    # Load API key
    env_path = ai_root / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not set in ai/.env")
        sys.exit(1)

    # ── Step 0: Show current balance ────────────────────────────────────
    print("\n[0/4] Current dataset balance:")
    before_counts = count_classes(data_root)
    print_balance("BEFORE (Current Data)", before_counts)

    # ── Step 1: Download v30 ────────────────────────────────────────────
    print(f"\n[1/4] Downloading v{ROBOFLOW_VERSION} from Roboflow...")
    download_dir = data_root / f"_temp_v{ROBOFLOW_VERSION}"
    if download_dir.exists():
        shutil.rmtree(download_dir)

    new_data_root = download_v30(api_key, download_dir)
    print(f"  Downloaded to: {new_data_root}")

    # ── Step 2: Build class mapping ─────────────────────────────────────
    print(f"\n[2/4] Building class mapping...")
    # Find data.yaml in the downloaded data
    data_yaml = None
    for candidate in [new_data_root / "data.yaml", download_dir / "data.yaml"]:
        if candidate.exists():
            data_yaml = candidate
            break

    if not data_yaml:
        # Search for it
        yamls = list(download_dir.rglob("data.yaml"))
        if yamls:
            data_yaml = yamls[0]
        else:
            print("ERROR: Could not find data.yaml in downloaded data")
            print("  Searched in:", download_dir)
            sys.exit(1)

    print(f"  Found data.yaml: {data_yaml}")

    # Show the class names in the new version
    with open(data_yaml, "r") as f:
        new_cfg = yaml.safe_load(f)
    print(f"  New version has {new_cfg.get('nc', '?')} classes:")
    names = new_cfg.get("names", [])
    if isinstance(names, dict):
        for idx, name in names.items():
            marker = " ← WE WANT THIS" if name in RAW_NAME_TO_TARGET_ID else ""
            print(f"    {idx}: {name}{marker}")
    else:
        for idx, name in enumerate(names):
            marker = " ← WE WANT THIS" if name in RAW_NAME_TO_TARGET_ID else ""
            print(f"    {idx}: {name}{marker}")

    mapping = build_mapping(data_yaml)

    if not mapping:
        print("ERROR: No matching classes found in the new version!")
        print("  The class names may have changed. Check the data.yaml above.")
        sys.exit(1)

    # ── Step 3: Merge ───────────────────────────────────────────────────
    print(f"\n[3/4] Merging new data into existing dataset...")
    added = merge_new_data(data_root, new_data_root, mapping)

    # ── Step 4: Clear cache and show results ────────────────────────────
    print(f"\n[4/4] Cleaning up...")
    clear_label_cache(data_root)

    # Remove temp download
    if download_dir.exists():
        shutil.rmtree(download_dir)
        print(f"  Cleaned up temp download")

    # Final report
    after_counts = count_classes(data_root)
    print_balance("AFTER (Merged Data)", after_counts)

    # Show improvement
    print(f"\n  Improvement Summary:")
    for cls in TARGET_CLASSES:
        before = sum(before_counts.get(s, {}).get(cls, 0) for s in ["train", "valid", "test"])
        after = sum(after_counts.get(s, {}).get(cls, 0) for s in ["train", "valid", "test"])
        diff = after - before
        emoji = "✅" if diff > 0 else "➖"
        print(f"    {emoji} {cls:<15}: {before:>4} → {after:>4} (+{diff})")

    # Check if we still need more data
    train_counts = after_counts.get("train", {})
    low_classes = [
        cls for cls in TARGET_CLASSES
        if train_counts.get(cls, 0) < 50
    ]
    if low_classes:
        print(f"\n  ⚠️  These classes still have <50 train labels: {', '.join(low_classes)}")
        print(f"     Consider manually annotating more images with Roboflow/CVAT.")
    else:
        print(f"\n  ✅ All classes have 50+ training labels!")

    print(f"\n{'=' * 60}")
    print(f"  [OK] Dataset augmentation complete!")
    print(f"  Next step: python ai/src/train.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
