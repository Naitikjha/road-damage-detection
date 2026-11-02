"""
Generate a synthetic RDD2022-like dataset for testing and verification.

Creates the exact PascalVOC XML and image folder structure for Japan and India subsets,
including positive pothole images, background-only images, and near-duplicate frames.
This allows verifying the entire pipeline (prepare_rdd.py, show_samples.py, etc.) immediately.
"""

import os
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def create_sample_image(width=600, height=600, potholes=None, noise_seed=None):
    """Generate a synthetic asphalt road image with road markings, horizon, and dark pothole patches."""
    if noise_seed is not None:
        np.random.seed(noise_seed)

    # Road background with gradient perspective and varied lighting
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Horizon line at top 25% of image
    horizon = int(height * 0.25)
    # Sky / background scenery
    sky_b = np.random.randint(180, 240)
    sky_g = np.random.randint(150, 210)
    sky_r = np.random.randint(130, 190)
    img[:horizon, :] = (sky_b, sky_g, sky_r)

    # Road asphalt with perspective gradient
    road_gray = np.random.randint(50, 85)
    for y in range(horizon, height):
        factor = 0.8 + 0.4 * (y - horizon) / (height - horizon)
        val = int(np.clip(road_gray * factor, 30, 110))
        img[y, :] = (val, val, val)

    # Add asphalt noise texture
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Perspective road lane markings
    lane_color = (255, 255, 255) if np.random.rand() > 0.5 else (0, 215, 255)
    pts_left = np.array([[width // 2 - 20, horizon], [width // 2 - 22, horizon], [40, height], [46, height]], np.int32)
    pts_right = np.array([[width // 2 + 20, horizon], [width // 2 + 22, horizon], [width - 40, height], [width - 46, height]], np.int32)
    cv2.fillPoly(img, [pts_left], lane_color)
    cv2.fillPoly(img, [pts_right], lane_color)

    # Center dashed marking
    for step in range(horizon + 20, height, 40):
        cv2.line(img, (width // 2, step), (width // 2, min(height, step + 20)), (255, 255, 255), 3)

    # Draw potholes as dark elliptical depressions with jagged edges
    if potholes:
        for xmin, ymin, xmax, ymax in potholes:
            cx = int((xmin + xmax) // 2)
            cy = int((ymin + ymax) // 2)
            axes = (max(5, int((xmax - xmin) // 2)), max(5, int((ymax - ymin) // 2)))
            cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (25, 25, 25), -1)
            cv2.ellipse(img, (cx, cy), (int(axes[0] * 0.7), int(axes[1] * 0.7)), 0, 0, 360, (15, 15, 15), -1)
            cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (110, 110, 110), 2)

    return img


def write_voc_xml(xml_path, filename, width, height, boxes):
    """Write PascalVOC format XML file."""
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = "images"
    fn = ET.SubElement(root, "filename")
    fn.text = filename

    size = ET.SubElement(root, "size")
    w = ET.SubElement(size, "width")
    w.text = str(width)
    h = ET.SubElement(size, "height")
    h.text = str(height)
    d = ET.SubElement(size, "depth")
    d.text = "3"

    for class_name, (xmin, ymin, xmax, ymax) in boxes:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_name
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        x1 = ET.SubElement(bndbox, "xmin")
        x1.text = str(xmin)
        y1 = ET.SubElement(bndbox, "ymin")
        y1.text = str(ymin)
        x2 = ET.SubElement(bndbox, "xmax")
        x2.text = str(xmax)
        y2 = ET.SubElement(bndbox, "ymax")
        y2.text = str(ymax)

    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def generate_dummy_dataset(target_dir="data/raw/RDD2022", num_per_country=30):
    """Generate dummy dataset for Japan and India subsets with known potholes and near-duplicates."""
    countries = ["Japan", "India"]

    for country in countries:
        base_dir = os.path.join(target_dir, country, "train")
        img_dir = os.path.join(base_dir, "images")
        xml_dir = os.path.join(base_dir, "annotations", "xmls")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(xml_dir, exist_ok=True)

        for i in range(num_per_country):
            img_name = f"{country}_{i:06d}.jpg"
            xml_name = f"{country}_{i:06d}.xml"
            img_path = os.path.join(img_dir, img_name)
            xml_path = os.path.join(xml_dir, xml_name)

            # Introduce intentional near-duplicate pair for i=5 and i=6
            if i == 6:
                prev_img_path = os.path.join(img_dir, f"{country}_{5:06d}.jpg")
                prev_img = cv2.imread(prev_img_path)
                # Slight noise to make it a realistic near-duplicate
                dup_img = np.clip(prev_img.astype(np.int16) + np.random.randint(-2, 3, prev_img.shape), 0, 255).astype(np.uint8)
                cv2.imwrite(img_path, dup_img)
                # Copy boxes from 5
                boxes = [("D40", (150, 220, 280, 310))]
                write_voc_xml(xml_path, img_name, 600, 600, boxes)
                continue

            # Some images have potholes (D40), some other damages (D00, D10), some background only
            boxes = []
            pothole_coords = []
            if i % 5 == 0:
                # Background only (no damage)
                pass
            elif i % 5 == 1:
                # Single pothole
                box = (150 + (i * 7) % 200, 200 + (i * 5) % 200, 280 + (i * 7) % 200, 310 + (i * 5) % 200)
                boxes.append(("D40", box))
                pothole_coords.append(box)
            elif i % 5 == 2:
                # Multiple potholes (dense scene test)
                b1 = (100, 250, 180, 320)
                b2 = (220, 300, 340, 420)
                b3 = (380, 180, 460, 250)
                boxes.extend([("D40", b1), ("D40", b2), ("D40", b3)])
                pothole_coords.extend([b1, b2, b3])
            elif i % 5 == 3:
                # Small pothole test (< 32x32 area)
                b_small = (200, 350, 225, 375)
                boxes.append(("D40", b_small))
                pothole_coords.append(b_small)
            else:
                # Other damage (e.g. longitudinal crack D00)
                boxes.append(("D00", (180, 100, 240, 450)))

            img = create_sample_image(600, 600, pothole_coords, noise_seed=i + (100 if country == "India" else 0))
            cv2.imwrite(img_path, img)
            write_voc_xml(xml_path, img_name, 600, 600, boxes)

    print(f"[+] Dummy RDD2022 dataset generated at: {os.path.abspath(target_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dummy RDD2022 dataset for pipeline testing")
    parser.add_argument("--target-dir", default="data/raw/RDD2022", help="Target root directory")
    parser.add_argument("--num-per-country", type=int, default=30, help="Number of images per country")
    args = parser.parse_args()
    generate_dummy_dataset(args.target_dir, args.num_per_country)
