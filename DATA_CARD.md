# DATA CARD: Road Damage Detection (RDD2022 Japan + India)

## 1. Dataset Overview and Provenance
* **Dataset Name**: Road Damage Dataset 2022 (RDD2022) - Japan & India Subsets
* **Official Repository**: [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
* **Citation**: 
  > Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). *Road Damage Detection Using Deep Neural Networks with Smartphone Images*. Computer-Aided Civil and Infrastructure Engineering.
* **Test Set Notice**: In the official RDD2022 challenge release, **ground truth annotations are public only for the training sets** of each country subset. Test sets do not have public labels. Therefore, all evaluation and tuning are conducted on strictly partitioned splits derived from the labeled data with near-duplicate separation.

---

## 2. Damage Class Taxonomy & Current Target
RDD2022 standardizes road damage into four primary classes:

| Class Code | Damage Name | Description | Default Status |
| :--- | :--- | :--- | :--- |
| **D00** | Longitudinal Crack | Cracks running parallel to the direction of vehicle travel | Optional (in `configs/all_classes.yaml`) |
| **D10** | Transverse Crack | Cracks running perpendicular to the direction of vehicle travel | Optional (in `configs/all_classes.yaml`) |
| **D20** | Alligator Crack | Interconnected polygonal cracks resembling reptile skin (fatigue) | Optional (in `configs/all_classes.yaml`) |
| **D40** | **Pothole** | Structural depressions or bowl-shaped road surface failures | **Primary Target** (in `configs/pothole_only.yaml`) |

In the default configuration (`configs/pothole_only.yaml`), **D40 (Pothole)** is mapped to `class_id: 0`.

---

## 3. Data Ingestion & Custom Preprocessing Pipeline

Our data preparation script (`tools/prepare_rdd.py`) enforces strict scientific standards to prevent data leakage and guarantee high recall:

### 3.1. Zero-Box-Drop Coverage Rule
* Many road safety systems fail because small, distant potholes are arbitrarily filtered out during dataset cleanup.
* **Rule**: All valid bounding boxes where $x_{min} < x_{max}$ and $y_{min} < y_{max}$ are preserved. Coordinates extending beyond image boundaries are safely clipped to $[0, W]$ and $[0, H]$. No small boxes are discarded.
* Box scale categorization follows COCO definitions:
  * **Small**: $\text{Area} < 32^2 = 1024 \text{ px}^2$
  * **Medium**: $32^2 \le \text{Area} < 96^2 = 9216 \text{ px}^2$
  * **Large**: $\text{Area} \ge 96^2 = 9216 \text{ px}^2$

### 3.2. Perceptual Hash (dHash) Deduplication
* **The Problem**: RDD2022 is collected using vehicle-mounted smartphones recording continuous video. When vehicles stop at traffic signals or move slowly in congestion, dozens of near-identical frames are captured. If split randomly, nearly identical frames would leak into both training and validation sets, producing artificially inflated metrics.
* **The Solution**: We compute a 64-bit Difference Hash (dHash) for each frame by resizing to $9 \times 8$ grayscale and comparing adjacent horizontal gradients ($\text{pixel}[x+1] > \text{pixel}[x]$).
* **Temporal Sequence Window**: Within a sequential window of 15 frames, any frame having a Hamming distance $\le 3$ bits against recent accepted frames is flagged as a temporal burst duplicate and excluded.

### 3.3. Background-Only Image Capping
* Dashcam datasets contain extensive stretches of undamaged road. While negative examples are necessary to prevent false positives, an overabundance creates extreme class sparsity.
* **Rule**: Background-only images (images containing 0 target damage annotations) are capped at a maximum of **10%** relative to positive images. They are selected deterministically using fixed seed 42.

### 3.4. Three-Way Split (Train / Tune / Val)
To honor the scientific requirement: *"Never tune thresholds on the test/val data used for the final report"*, the dataset is split into three separate subsets:
1. **Train Split (70%)**: Used solely for training model weights.
2. **Tune Split (15%)**: A dedicated split used solely to choose confidence thresholds, NMS IoU thresholds, and evaluate recall targets.
3. **Val Split (15%)**: Kept completely untouched during training and tuning; used solely to report final unbiased metrics (AP@0.5, AP@[.5:.95], PR curves, confusion matrix, coverage).

---

## 4. Custom Label Format Specification

Our custom format is clean, modular, and self-contained JSON stored in `data/processed/<config_name>/`:
* `train.json`, `tune.json`, `val.json`
* `metadata.json` (aggregate statistics, hash parameters, class counts)

### Sample Record (`train.json`):
```json
{
  "image_id": "Japan_000001",
  "file_name": "Japan_000001.jpg",
  "rel_path": "data/raw/RDD2022/Japan/train/images/Japan_000001.jpg",
  "abs_path": "D:/NAITIK/COLLEGE/ROAD-DAMAGE/data/raw/RDD2022/Japan/train/images/Japan_000001.jpg",
  "country": "Japan",
  "width": 600,
  "height": 600,
  "num_boxes": 2,
  "annotations": [
    {
      "class_id": 0,
      "class_name": "D40",
      "bbox": [152.0, 224.0, 281.0, 312.0],
      "area": 11352.0,
      "width": 129.0,
      "height": 88.0
    }
  ]
}
```

---

## 5. Current Counts & Verification Run

The table below reflects real numbers generated by running `tools/prepare_rdd.py` on the initialized dataset:

| Metric | Overall Pool | Train (70%) | Tune (15%) | Val (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Images** | 18 | 12 | 2 | 4 |
| **Positive Images (with Potholes)** | 17 | 11 | 2 | 4 |
| **Background-Only Images** | 1 | 1 | 0 | 0 |
| **Total Pothole (D40) Boxes** | 27 | 16 | 6 | 5 |
| **Small Boxes ($< 32^2$)** | 4 | 2 | 1 | 1 |
| **Medium Boxes ($32^2 - 96^2$)** | 10 | 6 | 2 | 2 |
| **Large Boxes ($\ge 96^2$)** | 13 | 8 | 3 | 2 |

*Note: The numbers above represent the verified dry-run test suite. When you extract the full RDD2022 Japan + India archive, re-running `python tools/prepare_rdd.py --config configs/pothole_only.yaml` will populate this table with the complete multi-thousand image counts.*

---

## 6. How to Download and Prepare the Full RDD2022 Dataset

1. Download `Japan.tar.gz` and `India.tar.gz` from the official repository:
   [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
2. Extract the archives into `data/raw/RDD2022/`:
   ```
   data/raw/RDD2022/
   ├── Japan/
   │   └── train/
   │       ├── images/
   │       └── annotations/xmls/
   └── India/
       └── train/
           ├── images/
           └── annotations/xmls/
   ```
3. Run the data preparation script:
   ```bash
   python tools/prepare_rdd.py --config configs/pothole_only.yaml
   ```
4. Verify sample annotations on 24 random images:
   ```bash
   python tools/show_samples.py --manifest data/processed/pothole_only/train.json --num-samples 24
   ```
