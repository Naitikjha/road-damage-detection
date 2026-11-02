"""
Prepare RDD2022 Dataset: Convert PascalVOC XML annotations to custom format.

Features:
1. Parses Japan and India subsets of RDD2022.
2. Filters to configurable target classes (default: D40 Pothole).
3. Preserves all valid bounding boxes (zero dropping of small boxes).
4. Runs perceptual hashing (dHash) to detect and exclude near-duplicate dashcam frames.
5. Caps background-only images (default <= 10% of positive images).
6. Splits into train (70%), tune (15%), and val (15%) splits with fixed seed 42.
   (The 'tune' split is reserved strictly for threshold and NMS tuning,
   preventing test/validation contamination).
7. Outputs train.json, tune.json, val.json, and metadata.json.
"""

import os
import sys
import glob
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import yaml
from tqdm import tqdm

# Ensure local packages in src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.phash import compute_dhash, hamming_distance, hash_to_hex


def find_files(country_dir: str) -> List[Tuple[str, str]]:
    """
    Locate corresponding image and XML files in a country directory.
    Handles various extraction folder layouts (train/images, train/annotations/xmls, etc.).
    """
    country_path = Path(country_dir)
    img_candidates = list(country_path.glob("**/*.jpg")) + list(country_path.glob("**/*.png"))

    paired = []
    for img_p in img_candidates:
        stem = img_p.stem
        # Common VOC XML locations in RDD2022
        possible_xmls = [
            img_p.parent.parent / "annotations" / "xmls" / f"{stem}.xml",
            img_p.parent.parent / "annotations" / f"{stem}.xml",
            img_p.parent.parent / "xmls" / f"{stem}.xml",
            img_p.parent / f"{stem}.xml",
        ]
        xml_match = None
        for cand in possible_xmls:
            if cand.exists():
                xml_match = cand
                break

        paired.append((str(img_p), str(xml_match) if xml_match else None))

    return paired


def parse_voc_xml(xml_path: str, target_classes: Dict[str, int]) -> Tuple[int, int, List[Dict[str, Any]], Dict[str, int]]:
    """
    Parse a single Pascal VOC XML file.

    Returns:
        width, height, annotations list, raw class count dictionary
    """
    counts = {}
    if not xml_path or not os.path.exists(xml_path):
        return 0, 0, [], counts

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_tag = root.find("size")
    width = int(float(size_tag.find("width").text)) if size_tag is not None and size_tag.find("width") is not None else 0
    height = int(float(size_tag.find("height").text)) if size_tag is not None and size_tag.find("height") is not None else 0

    annotations = []
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None:
            continue
        cls_name = name_tag.text.strip()
        counts[cls_name] = counts.get(cls_name, 0) + 1

        if cls_name not in target_classes:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (ValueError, TypeError, AttributeError):
            continue

        # Sanity check and coordinate clipping
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        if width > 0:
            xmax = min(float(width), xmax)
            xmin = min(xmin, xmax)
        if height > 0:
            ymax = min(float(height), ymax)
            ymin = min(ymin, ymax)

        w = xmax - xmin
        h = ymax - ymin

        # Keep all valid boxes with positive area (coverage requirement)
        if w > 0 and h > 0:
            annotations.append({
                "class_id": target_classes[cls_name],
                "class_name": cls_name,
                "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                "area": round(w * h, 2),
                "width": round(w, 2),
                "height": round(h, 2)
            })

    return width, height, annotations, counts


def categorize_box_size(area: float) -> str:
    """Categorize box area according to standard COCO metrics (32^2 = 1024, 96^2 = 9216)."""
    if area < 1024.0:
        return "small"
    elif area < 9216.0:
        return "medium"
    else:
        return "large"


def prepare_dataset(config_path: str):
    """Main preparation routine."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    raw_dir = Path(ds_cfg.get("raw_dir", "data/raw/RDD2022"))
    processed_dir = Path(ds_cfg.get("processed_dir", "data/processed/pothole_only"))
    countries = ds_cfg.get("countries", ["Japan", "India"])
    target_classes = ds_cfg.get("classes", {"D40": 0})
    class_names = ds_cfg.get("class_names", {0: "Pothole"})
    bg_ratio_cap = float(ds_cfg.get("background_ratio_cap", 0.10))
    splits = ds_cfg.get("splits", {"train": 0.70, "tune": 0.15, "val": 0.15})
    seed = int(ds_cfg.get("random_seed", 42))

    phash_cfg = ds_cfg.get("phash", {})
    phash_enabled = phash_cfg.get("enabled", True)
    hash_size = int(phash_cfg.get("hash_size", 8))
    max_hamming_dist = int(phash_cfg.get("max_hamming_distance", 3))
    window_size = phash_cfg.get("window_size", 15)

    random.seed(seed)

    print("=" * 70)
    print("ROAD DAMAGE DETECTION - DATA PREPARATION")
    print(f"Target classes: {target_classes}")
    print(f"Countries: {countries}")
    print(f"Source Directory: {raw_dir.resolve()}")
    print(f"Output Directory: {processed_dir.resolve()}")
    print(f"Random seed: {seed}")
    print(f"Near-duplicate filter: {phash_enabled} (Hamming dist <= {max_hamming_dist}, window={window_size})")
    print("=" * 70)

    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scan and parse all images & XMLs
    all_pairs = []
    for c in countries:
        c_dir = raw_dir / c
        if not c_dir.exists():
            print(f"[!] Warning: Country folder {c_dir} does not exist.")
            continue
        pairs = find_files(str(c_dir))
        print(f"[*] Found {len(pairs)} images in {c}")
        for img_p, xml_p in pairs:
            all_pairs.append((c, img_p, xml_p))

    if not all_pairs:
        raise FileNotFoundError(
            f"No images found in {raw_dir}. Please place RDD2022 Japan/India datasets in {raw_dir} "
            f"or run 'python tools/create_dummy_rdd.py' to generate a test set."
        )

    print(f"[*] Total candidate images found across all countries: {len(all_pairs)}")

    # Sort pairs to ensure deterministic processing in temporal order
    all_pairs.sort(key=lambda x: (x[0], x[1]))

    parsed_records = []
    raw_class_totals = {}
    duplicate_count = 0
    # Store recent hashes per country to eliminate temporal dashcam bursts
    recent_hashes_by_country: Dict[str, List] = {c: [] for c in countries}

    print("[*] Parsing XMLs and checking for near-duplicates via dHash...")
    for country, img_path, xml_path in tqdm(all_pairs, desc="Processing images"):
        w, h, annotations, raw_counts = parse_voc_xml(xml_path, target_classes)

        # Update raw class stats
        for k, v in raw_counts.items():
            raw_class_totals[k] = raw_class_totals.get(k, 0) + v

        # Read actual image dimensions if missing from XML
        if w <= 0 or h <= 0:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

        # Near-duplicate check with dHash
        if phash_enabled:
            try:
                curr_hash = compute_dhash(img_path, hash_size=hash_size)
                is_dup = False
                candidates = recent_hashes_by_country.get(country, [])
                for prev_hash in candidates:
                    if hamming_distance(curr_hash, prev_hash) <= max_hamming_dist:
                        is_dup = True
                        break
                if is_dup:
                    duplicate_count += 1
                    continue

                if country not in recent_hashes_by_country:
                    recent_hashes_by_country[country] = []
                recent_hashes_by_country[country].append(curr_hash)
                if window_size is not None and len(recent_hashes_by_country[country]) > window_size:
                    recent_hashes_by_country[country].pop(0)
            except Exception as e:
                pass

        rel_path = os.path.relpath(img_path, os.getcwd()).replace("\\", "/")
        record = {
            "image_id": Path(img_path).stem,
            "file_name": Path(img_path).name,
            "rel_path": rel_path,
            "country": country,
            "width": w,
            "height": h,
            "annotations": annotations,
            "num_boxes": len(annotations),
        }
        parsed_records.append(record)

    print(f"[*] Near-duplicate frames excluded by dHash: {duplicate_count}")
    print(f"[*] Retained frames after deduplication: {len(parsed_records)}")

    # 2. Separate positive vs background images and cap background
    positive_records = [r for r in parsed_records if r["num_boxes"] > 0]
    background_records = [r for r in parsed_records if r["num_boxes"] == 0]

    max_bg_allowed = int(len(positive_records) * bg_ratio_cap)
    random.shuffle(background_records)
    selected_bg = set(id(r) for r in background_records[:max_bg_allowed])

    # Keep temporal order for block assignment
    final_pool = [r for r in parsed_records if r["num_boxes"] > 0 or id(r) in selected_bg]

    print(f"[*] Positive images with target damage: {len(positive_records)}")
    print(f"[*] Available background-only images: {len(background_records)}")
    print(f"[*] Selected background images (capped at {bg_ratio_cap*100:.1f}%): {len(selected_bg)}")
    print(f"[*] Total images in dataset pool: {len(final_pool)}")

    # 3. Block-based split (group consecutive dashcam frames into blocks of block_size)
    target_block_size = int(splits.get("block_size", 50))
    # If pool is small (e.g. test runs), adapt block size so multiple blocks exist
    effective_block_size = target_block_size if len(final_pool) >= target_block_size * 3 else max(1, len(final_pool) // 6)

    # Group consecutive frames per country into blocks
    blocks: Dict[str, List[Dict[str, Any]]] = {}
    country_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in final_pool:
        country_groups.setdefault(r["country"], []).append(r)

    for country_name, records_in_country in country_groups.items():
        for idx, rec in enumerate(records_in_country):
            b_num = idx // effective_block_size
            b_id = f"{country_name}_blk_{b_num:04d}"
            rec["block_id"] = b_id
            blocks.setdefault(b_id, []).append(rec)

    block_id_list = list(blocks.keys())
    random.shuffle(block_id_list)

    n_blocks = len(block_id_list)
    n_train_b = max(1, int(n_blocks * splits["train"]))
    n_tune_b = max(1, int(n_blocks * splits["tune"])) if n_blocks >= 3 else 0
    n_val_b = n_blocks - n_train_b - n_tune_b
    if n_val_b <= 0 and n_blocks >= 3:
        n_train_b -= 1
        n_val_b = 1

    train_blocks = set(block_id_list[:n_train_b])
    tune_blocks = set(block_id_list[n_train_b : n_train_b + n_tune_b])
    val_blocks = set(block_id_list[n_train_b + n_tune_b :])

    # Strict assertion that no block appears in more than one split
    assert len(train_blocks.intersection(tune_blocks)) == 0, "Leakage: Block found in both train and tune splits!"
    assert len(train_blocks.intersection(val_blocks)) == 0, "Leakage: Block found in both train and val splits!"
    assert len(tune_blocks.intersection(val_blocks)) == 0, "Leakage: Block found in both tune and val splits!"

    train_data = [r for b in train_blocks for r in blocks[b]]
    tune_data = [r for b in tune_blocks for r in blocks[b]]
    val_data = [r for b in val_blocks for r in blocks[b]]

    print(f"[*] Block-based partitioning complete (block size: {effective_block_size} consecutive frames)")
    print(f"[*] Total blocks: {n_blocks} -> Train: {len(train_blocks)} blocks | Tune: {len(tune_blocks)} blocks | Val: {len(val_blocks)} blocks")

    # 4. Compute statistics
    def get_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats = {
            "num_images": len(records),
            "num_positive_images": sum(1 for r in records if r["num_boxes"] > 0),
            "num_background_images": sum(1 for r in records if r["num_boxes"] == 0),
            "total_boxes": sum(r["num_boxes"] for r in records),
            "boxes_by_class": {},
            "boxes_by_size": {"small": 0, "medium": 0, "large": 0},
            "boxes_per_image": {
                "max": max((r["num_boxes"] for r in records), default=0),
                "avg": round(sum(r["num_boxes"] for r in records) / max(1, len(records)), 2),
            },
        }
        for r in records:
            for ann in r["annotations"]:
                cname = ann["class_name"]
                stats["boxes_by_class"][cname] = stats["boxes_by_class"].get(cname, 0) + 1
                size_cat = categorize_box_size(ann["area"])
                stats["boxes_by_size"][size_cat] += 1
        return stats

    stats_train = get_stats(train_data)
    stats_tune = get_stats(tune_data)
    stats_val = get_stats(val_data)
    stats_overall = get_stats(final_pool)

    metadata = {
        "dataset_name": "RDD2022_Custom",
        "random_seed": seed,
        "target_classes": target_classes,
        "class_names": class_names,
        "near_duplicate_filter": {
            "enabled": phash_enabled,
            "hash_size": hash_size,
            "max_hamming_distance": max_hamming_dist,
            "window_size": window_size,
            "excluded_count": duplicate_count,
        },
        "background_cap_ratio": bg_ratio_cap,
        "splits_ratio": splits,
        "block_partitioning": {
            "block_size": effective_block_size,
            "total_blocks": n_blocks,
            "train_blocks": len(train_blocks),
            "tune_blocks": len(tune_blocks),
            "val_blocks": len(val_blocks),
        },
        "raw_classes_found_in_xmls": raw_class_totals,
        "counts": {
            "overall": stats_overall,
            "train": stats_train,
            "tune": stats_tune,
            "val": stats_val,
        },
    }

    # 5. Save json manifests
    for name, data in [("train.json", train_data), ("tune.json", tune_data), ("val.json", val_data)]:
        out_p = processed_dir / name
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[+] Wrote {out_p} ({len(data)} images)")

    meta_p = processed_dir / "metadata.json"
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Wrote {meta_p}")

    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print(f"Train split: {stats_train['num_images']} images | {stats_train['total_boxes']} target boxes")
    print(f"Tune split:  {stats_tune['num_images']} images | {stats_tune['total_boxes']} target boxes")
    print(f"Val split:   {stats_val['num_images']} images | {stats_val['total_boxes']} target boxes")
    print(f"Size breakdown (Overall): {stats_overall['boxes_by_size']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RDD2022 dataset for anchor-free detector")
    parser.add_argument("--config", default="configs/pothole_only.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    prepare_dataset(args.config)
