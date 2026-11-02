"""
Road Damage PyTorch Dataset with Letterbox Resize.

Loads custom JSON format produced by tools/prepare_rdd.py.
Applies aspect-ratio preserving letterboxing to input size (e.g. 512x512).
Converts annotations to tensors with adjusted box coordinates.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox_image(
    image: np.ndarray,
    target_size: int = 512,
    fill_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with aspect-ratio preserving padding (letterbox).

    Args:
        image: Original BGR/RGB image array (H, W, C).
        target_size: Target square dimension (e.g. 512).
        fill_value: Pixel value for padding (114 is neutral gray).

    Returns:
        padded_img: Letterboxed image of shape (target_size, target_size, C).
        scale: Resize scale factor applied.
        pad: (pad_x, pad_y) offsets added to top/left.
    """
    orig_h, orig_w = image.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    canvas = np.full((target_size, target_size, 3), fill_value, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def adjust_boxes_for_letterbox(
    boxes: np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    target_size: int = 512,
) -> np.ndarray:
    """
    Adjust [xmin, ymin, xmax, ymax] boxes to letterboxed coordinates.

    Args:
        boxes: NumPy array of shape (N, 4) in original pixel coordinates.
        scale: Scale factor from letterbox_image.
        pad: (pad_x, pad_y) from letterbox_image.
        target_size: Max boundary to clip against.

    Returns:
        NumPy array of shape (N, 4) in letterboxed pixel coordinates.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    pad_x, pad_y = pad
    adjusted = boxes.copy().astype(np.float32)
    adjusted[:, [0, 2]] = adjusted[:, [0, 2]] * scale + pad_x
    adjusted[:, [1, 3]] = adjusted[:, [1, 3]] * scale + pad_y

    # Clip to valid range [0, target_size]
    adjusted[:, [0, 2]] = np.clip(adjusted[:, [0, 2]], 0, target_size)
    adjusted[:, [1, 3]] = np.clip(adjusted[:, [1, 3]], 0, target_size)

    return adjusted


class RoadDamageDataset(Dataset):
    """
    PyTorch Dataset loading processed JSON manifests for road damage detection.
    """

    def __init__(
        self,
        manifest_path: str,
        input_size: int = 512,
        transform: Optional[Any] = None,
        is_training: bool = False,
    ):
        """
        Args:
            manifest_path: Path to train.json, tune.json, or val.json.
            input_size: Target square resolution (default 512).
            transform: Optional box-aware augmentation pipeline.
            is_training: Whether dataset is used for training mode.
        """
        self.manifest_path = manifest_path
        self.input_size = input_size
        self.transform = transform
        self.is_training = is_training

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.records[idx]
        img_path = record.get("abs_path")
        if not img_path or not os.path.exists(img_path):
            img_path = str(Path(record["rel_path"]).resolve())

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        # Letterbox image
        letterbox_img, scale, pad = letterbox_image(img_bgr, target_size=self.input_size)

        # Parse boxes and classes
        raw_boxes = []
        raw_classes = []
        for ann in record.get("annotations", []):
            raw_boxes.append(ann["bbox"])
            raw_classes.append(ann["class_id"])

        if len(raw_boxes) > 0:
            boxes_np = np.array(raw_boxes, dtype=np.float32)
            classes_np = np.array(raw_classes, dtype=np.int64)
            boxes_adj = adjust_boxes_for_letterbox(boxes_np, scale, pad, target_size=self.input_size)
            # Combine into (N, 5): [xmin, ymin, xmax, ymax, class_id]
            targets = np.hstack([boxes_adj, classes_np[:, None]])
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        # Convert image to RGB float tensor (3, H, W) normalized to [0, 1]
        img_rgb = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(targets).float()

        meta = {
            "image_id": record.get("image_id"),
            "file_name": record.get("file_name"),
            "orig_size": (record.get("width"), record.get("height")),
            "scale": scale,
            "pad": pad,
        }

        return img_tensor, target_tensor, meta


def collate_fn(batch):
    """Custom collator handling variable number of target boxes per image."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    return images, targets, metas
