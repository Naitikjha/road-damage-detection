"""
Visualize Ground Truth Annotations on Sample Images.

Loads 24 random images from a processed split (or raw dataset),
draws ground truth bounding boxes with class tags and bounding dimensions,
and saves a consolidated 4x6 inspection grid.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

# Distinct colors for classes (BGR)
CLASS_COLORS = [
    (0, 165, 255),   # Orange: Pothole (D40)
    (255, 50, 50),   # Blue: Longitudinal Crack (D00)
    (50, 205, 50),   # Lime Green: Transverse Crack (D10)
    (180, 50, 220),  # Purple: Alligator Crack (D20)
]


def draw_box(img: np.ndarray, bbox: List[float], label: str, color: Tuple[int, int, int], thickness: int = 2):
    """Draw an annotated bounding box with readable background label badge."""
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

    # Label badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Place label above box if space permits, else inside
    label_ymin = max(ymin - text_h - 4, 0)
    label_ymax = label_ymin + text_h + 4
    label_xmax = min(xmin + text_w + 6, img.shape[1])

    cv2.rectangle(img, (xmin, label_ymin), (label_xmax, label_ymax), color, -1)
    cv2.putText(
        img,
        label,
        (xmin + 3, label_ymin + text_h + 1),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        lineType=cv2.LINE_AA,
    )


def render_sample(record: Dict[str, Any], target_size: Tuple[int, int] = (400, 400)) -> np.ndarray:
    """Read an image record, draw its annotations, and resize for grid display."""
    img_path = record.get("abs_path") or record.get("rel_path")
    if not os.path.exists(img_path):
        # Try finding it relative to current working directory
        img_path = str(Path(record["rel_path"]).resolve())

    img = cv2.imread(img_path)
    if img is None:
        # Create fallback error placeholder
        placeholder = np.full((target_size[1], target_size[0], 3), 40, dtype=np.uint8)
        cv2.putText(placeholder, "File Not Found", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return placeholder

    orig_h, orig_w = img.shape[:2]
    annotated = img.copy()

    annotations = record.get("annotations", [])
    if len(annotations) == 0:
        # Mark as background
        cv2.putText(
            annotated,
            "[Background - Negative]",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
            lineType=cv2.LINE_AA,
        )
    else:
        for ann in annotations:
            cls_id = ann.get("class_id", 0)
            cls_name = ann.get("class_name", "Damage")
            bbox = ann["bbox"]
            w = ann.get("width", bbox[2] - bbox[0])
            h = ann.get("height", bbox[3] - bbox[1])
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            label = f"{cls_name} ({int(w)}x{int(h)})"
            draw_box(annotated, bbox, label, color, thickness=2)

    # Add header banner with Image ID and Country
    img_id = record.get("image_id", Path(img_path).stem)
    country = record.get("country", "")
    boxes_count = len(annotations)
    header_text = f"{img_id} | {country} | {boxes_count} box(es)"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        header_text,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    # Resize to target tile size
    resized = cv2.resize(annotated, target_size, interpolation=cv2.INTER_AREA)
    return resized


def build_grid(
    manifest_path: str,
    output_path: str = "outputs/sample_inspections.jpg",
    num_samples: int = 24,
    cols: int = 6,
    seed: int = 42,
    save_individual: bool = False,
):
    """Load sample images, build grid collage, and save to output."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}. Run tools/prepare_rdd.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print("[!] Warning: Manifest is empty.")
        return

    random.seed(seed)
    # Pick up to num_samples (prioritize images with boxes, include some background if available)
    positives = [r for r in records if r.get("num_boxes", 0) > 0]
    backgrounds = [r for r in records if r.get("num_boxes", 0) == 0]

    selected = []
    if len(positives) >= num_samples:
        # Sample mainly positives with 1-2 backgrounds
        num_bg = min(len(backgrounds), max(1, num_samples // 10))
        num_pos = num_samples - num_bg
        selected = random.sample(positives, num_pos) + random.sample(backgrounds, num_bg)
    else:
        # If fewer than num_samples, take all available records
        selected = list(records)
        while len(selected) < num_samples and len(selected) > 0:
            selected.append(random.choice(records))

    random.shuffle(selected)
    selected = selected[:num_samples]

    tile_size = (360, 360)
    rendered_tiles = [render_sample(rec, tile_size) for rec in selected]

    # Calculate grid dimensions
    rows = (len(rendered_tiles) + cols - 1) // cols
    grid_h = rows * tile_size[1]
    grid_w = cols * tile_size[0]
    grid_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, tile in enumerate(rendered_tiles):
        r = idx // cols
        c = idx % cols
        y1 = r * tile_size[1]
        y2 = y1 + tile_size[1]
        x1 = c * tile_size[0]
        x2 = x1 + tile_size[0]
        grid_canvas[y1:y2, x1:x2] = tile

    # Ensure output directory exists
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(output_path, grid_canvas)
    print(f"[+] Successfully saved {num_samples}-sample inspection grid ({rows}x{cols}) to:")
    print(f"    {os.path.abspath(output_path)}")

    if save_individual:
        indiv_dir = out_dir / "individual_samples"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for idx, (rec, tile) in enumerate(zip(selected, rendered_tiles)):
            img_id = rec.get("image_id", f"sample_{idx:02d}")
            cv2.imwrite(str(indiv_dir / f"{img_id}.jpg"), tile)
        print(f"[+] Saved individual samples to: {indiv_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ground truth annotations on sample images")
    parser.add_argument(
        "--manifest",
        default="data/processed/pothole_only/train.json",
        help="Path to processed JSON manifest (train.json, val.json, tune.json)",
    )
    parser.add_argument(
        "--output",
        default="outputs/sample_inspections.jpg",
        help="Path to output inspection grid image",
    )
    parser.add_argument("--num-samples", type=int, default=24, help="Number of sample images to display")
    parser.add_argument("--cols", type=int, default=6, help="Number of columns in visualization grid")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--save-individual", action="store_true", help="Also save individual annotated images")
    args = parser.parse_args()

    build_grid(
        manifest_path=args.manifest,
        output_path=args.output,
        num_samples=args.num_samples,
        cols=args.cols,
        seed=args.seed,
        save_individual=args.save_individual,
    )
