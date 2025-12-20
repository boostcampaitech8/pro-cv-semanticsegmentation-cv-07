import os
import json
import argparse
from glob import glob
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
SMALL_CLASSES = [
    "Pisiform",
    "Trapezium",
    "Trapezoid",
    "finger-19",
    "finger-20",
    "finger-21",
    "finger-22",
    "finger-23",
    "finger-24",
]


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build image-level metadata for hand bone segmentation"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Path to data/train/DCM"
    )
    parser.add_argument(
        "--json_root",
        type=str,
        required=True,
        help="Path to data/train/outputs_json"
    )
    parser.add_argument(
        "--flags_csv",
        type=str,
        required=True,
        help="Path to manually annotated flags.csv"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output metadata csv path"
    )
    return parser.parse_args()


# -------------------------------------------------
# Collect image-json pairs
# -------------------------------------------------
def collect_image_pairs(image_root, json_root):
    rows = []

    for sid in sorted(os.listdir(image_root)):
        img_dir = os.path.join(image_root, sid)
        json_dir = os.path.join(json_root, sid)

        if not os.path.isdir(img_dir) or not os.path.isdir(json_dir):
            continue

        for img_path in glob(os.path.join(img_dir, "*.png")):
            image_id = os.path.basename(img_path)
            json_path = os.path.join(
                json_dir, image_id.replace(".png", ".json")
            )

            if not os.path.exists(json_path):
                continue

            rows.append({
                "subject_id": sid,
                "image_id": image_id,
                "image_path": img_path,
                "json_path": json_path
            })

    return pd.DataFrame(rows)


# -------------------------------------------------
# Annotation utilities
# -------------------------------------------------
def load_polygons(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    polygons = []
    for ann in data.get("annotations", []):
        cls = ann["label"]
        pts = np.array(ann["points"], dtype=np.int32)
        polygons.append((cls, pts))

    return polygons


def rasterize_masks(polygons, h, w):
    masks = {}
    for cls, pts in polygons:
        if cls not in masks:
            masks[cls] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(masks[cls], [pts], 1)
    return masks


# -------------------------------------------------
# Difficulty-related features
# -------------------------------------------------
def compute_overlap_pixels(masks):
    if len(masks) <= 1:
        return 0
    stack = np.stack(list(masks.values()), axis=0)
    return int((stack.sum(axis=0) >= 2).sum())


def compute_boundary_adj_pixels(masks):
    if len(masks) <= 1:
        return 0

    kernel = np.ones((3, 3), np.uint8)
    edges = []

    for m in masks.values():
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
        edges.append(edge)

    stack = np.stack(edges, axis=0)
    return int((stack.sum(axis=0) >= 2).sum())


def compute_small_mask_outlier_count(masks, thresholds):
    cnt = 0
    for cls, m in masks.items():
        if m.sum() < thresholds.get(cls, 0):
            cnt += 1
    return cnt


def compute_total_bone_pixels(masks):
    return int(sum(m.sum() for m in masks.values()))


def compute_small_class_stats(masks):
    small_pixels = [
        int(m.sum()) for cls, m in masks.items() if cls in SMALL_CLASSES
    ]

    if len(small_pixels) == 0:
        return 0, 0

    return sum(small_pixels), min(small_pixels)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    pair_df = collect_image_pairs(args.image_root, args.json_root)
    if len(pair_df) == 0:
        raise RuntimeError("No valid image-json pairs found.")

    # ---------- Pass 1: class-wise pixel statistics ----------
    class_pixels = defaultdict(list)
    temp_records = []

    for _, row in tqdm(
        pair_df.iterrows(),
        total=len(pair_df),
        desc="Pass 1: collecting mask statistics"
    ):
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        polygons = load_polygons(row["json_path"])
        masks = rasterize_masks(polygons, h, w)

        for cls, m in masks.items():
            class_pixels[cls].append(int(m.sum()))

        temp_records.append({
            **row,
            "masks": masks
        })

    # class-wise small-mask threshold (5th percentile)
    thresholds = {
        cls: np.percentile(pixels, 5)
        for cls, pixels in class_pixels.items()
        if len(pixels) > 0
    }

    # ---------- Pass 2: build metadata ----------
    rows = []
    for r in tqdm(temp_records, desc="Pass 2: building metadata"):
        masks = r["masks"]

        total_bone_pixels = compute_total_bone_pixels(masks)
        small_class_pixels, min_small_class_pixels = compute_small_class_stats(masks)

        rows.append({
            "subject_id": r["subject_id"],
            "image_id": r["image_id"],
            "image_path": r["image_path"],
            "json_path": r["json_path"],

            # global structure signals
            "overlap_pixels": compute_overlap_pixels(masks),
            "boundary_adj_pixels": compute_boundary_adj_pixels(masks),
            "small_mask_outlier_count": compute_small_mask_outlier_count(
                masks, thresholds
            ),

            # pixel statistics
            "total_bone_pixels": total_bone_pixels,
            "small_class_pixels": small_class_pixels,
            "min_small_class_pixels": min_small_class_pixels,
            "small_class_pixel_ratio": (
                small_class_pixels / total_bone_pixels
                if total_bone_pixels > 0 else 0.0
            ),
        })

    meta_df = pd.DataFrame(rows)

    # ---------- Merge manual flags ----------
    flags_df = pd.read_csv(args.flags_csv)

    meta_df = meta_df.merge(
        flags_df,
        on=["subject_id", "image_id"],
        how="left"
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta_df.to_csv(args.out, index=False)

    print(f"[OK] Metadata saved to: {args.out}")
    print(f"Total rows: {len(meta_df)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)