"""
Unit tests for Phase 1: Data Preparation, Perceptual Hashing, and Dataset Loading.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.phash import compute_dhash, hamming_distance, hash_to_hex, hex_to_hash
from src.data.dataset import letterbox_image, adjust_boxes_for_letterbox, RoadDamageDataset, collate_fn
from tools.prepare_rdd import parse_voc_xml, categorize_box_size


def test_dhash_identical_and_different():
    """Verify dHash produces 0 distance for identical images and >0 for different images."""
    img1 = np.full((100, 100, 3), 120, dtype=np.uint8)
    img1[20:80, 20:80] = 30  # square patch

    img_same = img1.copy()
    img_diff = np.full((100, 100, 3), 200, dtype=np.uint8)

    h1 = compute_dhash(img1, hash_size=8)
    h_same = compute_dhash(img_same, hash_size=8)
    h_diff = compute_dhash(img_diff, hash_size=8)

    assert len(h1) == 64
    assert hamming_distance(h1, h_same) == 0
    # Different images should have different hashes
    assert hamming_distance(h1, h_diff) > 0


def test_dhash_hex_serialization():
    """Verify hash to hex conversion and back is lossless."""
    hash_arr = np.random.choice([True, False], size=64)
    hex_str = hash_to_hex(hash_arr)
    recovered = hex_to_hash(hex_str, hash_size=8)
    assert np.array_equal(hash_arr, recovered)


def test_parse_voc_xml(tmp_path):
    """Test PascalVOC XML parser extracting target classes, clipping, and discarding empty boxes."""
    xml_content = """<annotation>
        <size><width>600</width><height>600</height><depth>3</depth></size>
        <object>
            <name>D40</name>
            <bndbox><xmin>100</xmin><ymin>150</ymin><xmax>250</xmax><ymax>300</ymax></bndbox>
        </object>
        <object>
            <name>D00</name>
            <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>80</xmax><ymax>200</ymax></bndbox>
        </object>
        <object>
            <name>D40</name>
            <bndbox><xmin>550</xmin><ymin>550</ymin><xmax>650</xmax><ymax>650</ymax></bndbox>
        </object>
    </annotation>"""
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    # Target class: pothole only (D40)
    w, h, anns, raw_counts = parse_voc_xml(str(xml_file), target_classes={"D40": 0})

    assert w == 600
    assert h == 600
    assert raw_counts == {"D40": 2, "D00": 1}
    # Only 2 D40 annotations should be kept
    assert len(anns) == 2
    # Verify clipping of the second D40 box: xmax, ymax must not exceed 600
    assert anns[1]["bbox"][2] == 600.0
    assert anns[1]["bbox"][3] == 600.0


def test_categorize_box_size():
    """Verify box sizing bins (COCO standards: <32^2 small, 32^2-96^2 medium, >=96^2 large)."""
    assert categorize_box_size(500.0) == "small"
    assert categorize_box_size(4000.0) == "medium"
    assert categorize_box_size(15000.0) == "large"


def test_letterbox_and_box_adjustment():
    """Verify letterboxing preserves aspect ratio and adjusts box coordinates without distortion."""
    # Test a wide 800x400 image into 512x512
    orig_img = np.zeros((400, 800, 3), dtype=np.uint8)
    target_dim = 512
    letterboxed, scale, pad = letterbox_image(orig_img, target_size=target_dim)

    assert letterboxed.shape == (target_dim, target_dim, 3)
    # Scale should be 512 / 800 = 0.64
    assert np.isclose(scale, 0.64, atol=1e-2)
    # Width fits exactly (pad_x=0), height has padding top and bottom
    assert pad[0] == 0
    assert pad[1] > 0

    # Adjust a box at [200, 100, 400, 300]
    boxes = np.array([[200.0, 100.0, 400.0, 300.0]], dtype=np.float32)
    adj_boxes = adjust_boxes_for_letterbox(boxes, scale, pad, target_size=target_dim)

    # Check scaled dimensions
    orig_box_w = 400.0 - 200.0
    adj_box_w = adj_boxes[0, 2] - adj_boxes[0, 0]
    assert np.isclose(adj_box_w, orig_box_w * scale, atol=1.0)


def test_dataset_and_dataloader():
    """Verify PyTorch RoadDamageDataset and DataLoader work on processed manifest."""
    manifest_path = "data/processed/pothole_only/train.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Manifest train.json does not exist yet. Run prepare_rdd.py first.")

    dataset = RoadDamageDataset(manifest_path=manifest_path, input_size=512)
    assert len(dataset) > 0

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    images, targets, metas = next(iter(loader))

    assert images.shape == (2, 3, 512, 512)
    assert isinstance(targets, list)
    assert len(targets) == 2
    assert isinstance(metas, list)
    assert len(metas) == 2
