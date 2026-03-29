"""
Download 100 COCO 2017 validation images containing vehicles or pedestrians.

Usage:
    python datasets/download_coco.py

Output:
    datasets/coco_val/images/      — 100 JPEG images
    datasets/coco_val/annotations.json — filtered ground truth annotations
"""

import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# COCO 2017 validation annotation URL
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"

# ADAS-relevant COCO category IDs
ADAS_CATEGORY_IDS = {1, 3, 6, 8}  # person, car, bus, truck

TARGET_COUNT = 100

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "coco_val"
IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_DIR = OUTPUT_DIR
ANNOTATIONS_ZIP = BASE_DIR / "annotations_trainval2017.zip"
ANNOTATIONS_JSON = BASE_DIR / "annotations_raw" / "instances_val2017.json"
OUTPUT_ANNOTATIONS = OUTPUT_DIR / "annotations.json"


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"Downloading {desc or url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def ensure_annotations() -> dict:
    """Download and extract COCO annotations if not already present."""
    if not ANNOTATIONS_JSON.exists():
        download_file(ANNOTATIONS_URL, ANNOTATIONS_ZIP, "COCO annotations archive")
        print("Extracting annotations ...")
        with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as zf:
            zf.extract("annotations/instances_val2017.json", BASE_DIR / "annotations_raw")
        # ZipFile extracts with subdirectory structure; move to expected path
        extracted = BASE_DIR / "annotations_raw" / "annotations" / "instances_val2017.json"
        if extracted.exists() and not ANNOTATIONS_JSON.exists():
            ANNOTATIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
            extracted.rename(ANNOTATIONS_JSON)
        print("Extraction complete.")

    print("Loading COCO annotations ...")
    with open(ANNOTATIONS_JSON) as f:
        return json.load(f)


def select_images(coco_data: dict, count: int = TARGET_COUNT) -> list:
    """Return up to `count` image metadata dicts that contain ADAS categories."""
    # Build set of image IDs that have at least one ADAS annotation
    adas_image_ids = set()
    for ann in coco_data["annotations"]:
        if ann["category_id"] in ADAS_CATEGORY_IDS:
            adas_image_ids.add(ann["image_id"])

    # Build image_id -> image_info mapping
    id_to_info = {img["id"]: img for img in coco_data["images"]}

    # Stable selection: sorted by image ID for reproducibility
    selected_ids = sorted(adas_image_ids)[:count]
    return [id_to_info[img_id] for img_id in selected_ids if img_id in id_to_info]


def download_images(selected_images: list) -> None:
    """Download images that aren't already present."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(selected_images)} images ...")
    for img_info in tqdm(selected_images, unit="img"):
        filename = img_info["file_name"]
        dest = IMAGES_DIR / filename
        if dest.exists():
            continue
        url = f"{IMAGE_BASE_URL}/{filename}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except requests.RequestException as e:
            print(f"  Warning: failed to download {filename}: {e}", file=sys.stderr)


def build_filtered_annotations(coco_data: dict, selected_images: list) -> dict:
    """
    Build a filtered annotation dict with only the selected images and
    their ADAS-category annotations.
    """
    selected_ids = {img["id"] for img in selected_images}

    filtered_annotations = [
        ann for ann in coco_data["annotations"]
        if ann["image_id"] in selected_ids and ann["category_id"] in ADAS_CATEGORY_IDS
    ]

    filtered_categories = [
        cat for cat in coco_data["categories"]
        if cat["id"] in ADAS_CATEGORY_IDS
    ]

    return {
        "images": selected_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories,
    }


def main():
    print("=== COCO Dataset Downloader for ADAS ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get annotations
    coco_data = ensure_annotations()
    print(f"Loaded {len(coco_data['images'])} images, "
          f"{len(coco_data['annotations'])} annotations from COCO val2017.\n")

    # Step 2: Select 100 ADAS images
    selected = select_images(coco_data, TARGET_COUNT)
    print(f"Selected {len(selected)} images containing persons/vehicles.\n")

    # Step 3: Download images
    download_images(selected)
    actual_count = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"\nImages saved to: {IMAGES_DIR}  ({actual_count} files)")

    # Step 4: Save filtered annotations
    filtered = build_filtered_annotations(coco_data, selected)
    OUTPUT_ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ANNOTATIONS, "w") as f:
        json.dump(filtered, f)
    print(f"Annotations saved to: {OUTPUT_ANNOTATIONS}")
    print(f"  {len(filtered['annotations'])} ground-truth boxes across "
          f"{len(filtered['images'])} images.")
    print("\nDone.")


if __name__ == "__main__":
    main()
