# DATA CARD: Road Damage Detection (RDD2022 Japan + India)

## 1. Dataset Overview and Provenance
* **Dataset Name**: Road Damage Dataset 2022 (RDD2022) - Japan & India Subsets
* **Official Repository**: [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
* **Citations**:
  > Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2022). *Crowdsensing-based Road Damage Detection: An Extended Dataset and Transferability Analysis*. arXiv:2208.14444.
  > 
  > Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). *Road Damage Detection Using Deep Neural Networks with Smartphone Images*. Computer-Aided Civil and Infrastructure Engineering.
* **Test Set Notice**: In the official RDD2022 release, **ground truth annotations are public only for the training sets** of each country subset. Challenge test sets do not have public labels. Therefore, all evaluation and tuning are conducted on strictly partitioned splits derived from the labeled data with near-duplicate separation and block-based grouping.

---

## 2. Damage Class Taxonomy & Current Target
RDD2022 standardizes road damage into four primary classes:

| Class Code | Damage Name | Description | Default Status |
| :--- | :--- | :--- | :--- |
| **D00** | Longitudinal Crack | Cracks running parallel to vehicle travel direction | Optional (in `configs/all_classes.yaml`) |
| **D10** | Transverse Crack | Cracks running perpendicular to vehicle travel direction | Optional (in `configs/all_classes.yaml`) |
| **D20** | Alligator Crack | Interconnected polygonal fatigue cracks | Optional (in `configs/all_classes.yaml`) |
| **D40** | **Pothole** | Structural depressions or bowl-shaped road failures | **Primary Target** (in `configs/pothole_only.yaml`) |

In the default configuration (`configs/pothole_only.yaml`), **D40 (Pothole)** is mapped to `class_id: 0`.

---

## 3. Data Ingestion & Preprocessing Pipeline

Our data preparation script (`tools/prepare_rdd.py`) enforces strict standards to guarantee high coverage and zero data leakage:

### 3.1. Zero-Box-Drop Coverage Rule
* Small and distant potholes are critical for early road hazard intervention.
* **Rule**: All valid bounding boxes where $x_{\min} < x_{\max}$ and $y_{\min} < y_{\max}$ are preserved. Coordinates extending beyond image boundaries are clipped to $[0, W]$ and $[0, H]$. No small boxes are discarded during preprocessing.
* Standard COCO scale categorization:
  * **Small**: $\text{Area} < 32^2 = 1024 \text{ px}^2$
  * **Medium**: $32^2 \le \text{Area} < 96^2 = 9216 \text{ px}^2$
  * **Large**: $\text{Area} \ge 96^2 = 9216 \text{ px}^2$

### 3.2. Perceptual Hash (dHash) Deduplication
* Continuous dashcam recording produces bursts of nearly identical frames when vehicles are stationary or moving slowly.
* We compute a 64-bit Difference Hash (`dHash`) by resizing to $9 \times 8$ grayscale and comparing adjacent horizontal gradients.
* Frames with Hamming distance $\le 3$ within a temporal window of 15 frames are flagged and excluded.

### 3.3. Background-Only Image Capping
* Negative frames (zero target damage annotations) are capped at a maximum of **10%** relative to positive frames to prevent extreme sparsity.

### 3.4. Block-Based Split (Train / Tune / Val)
* Rather than a naive random shuffle by image (which can leak temporally adjacent video clips), we group consecutive frames into **contiguous blocks of 50 frames**.
* Whole blocks are assigned to:
  1. **Train Split (70%)**: Model weight optimization.
  2. **Tune Split (15%)**: Dedicated threshold and NMS parameter tuning (strictly separated from evaluation).
  3. **Val Split (15%)**: Final unbiased evaluation.
* **Leakage Assertion**: The split pipeline enforces a hard programmatic assertion confirming that zero blocks overlap across splits:
  $$\text{Blocks}_{\text{train}} \cap \text{Blocks}_{\text{tune}} = \emptyset, \quad \text{Blocks}_{\text{train}} \cap \text{Blocks}_{\text{val}} = \emptyset, \quad \text{Blocks}_{\text{tune}} \cap \text{Blocks}_{\text{val}} = \emptyset$$

---

## 4. Custom Label Format Specification

Stored in `data/processed/<config_name>/` using clean JSON manifests with **relative paths only** (no machine-specific absolute paths):

### Sample Record:
```json
{
  "image_id": "Japan_000001",
  "file_name": "Japan_000001.jpg",
  "rel_path": "data/raw/RDD2022/Japan/train/images/Japan_000001.jpg",
  "country": "Japan",
  "block_id": "Japan_blk_0000",
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

## 5. Pipeline Dry-Run Verification Counts (Synthetic Dummy Data)

> **[IMPORTANT]**
> The counts in the table below were generated from a **synthetic dummy dry-run** created by `tools/create_dummy_rdd.py` to verify code correctness, parsing, block-based splitting, and PyTorch dataset loading.
> 
> Real dataset numbers will be generated directly from the real RDD2022 Japan + India archives when `tools/prepare_rdd.py` is executed on your downloaded data. All processed JSONs and dummy visualization images are untracked via `.gitignore` to prevent committing mock data to the repository.

| Metric | Overall Pool (Dummy) | Train (70%) | Tune (15%) | Val (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Images** | 18 | 12 | 2 | 4 |
| **Positive Images (with Potholes)** | 17 | 11 | 2 | 4 |
| **Background-Only Images** | 1 | 1 | 0 | 0 |
| **Total Pothole (D40) Boxes** | 27 | 17 | 4 | 6 |
| **Small Boxes ($< 32^2$)** | 4 | 2 | 1 | 1 |
| **Medium Boxes ($32^2 - 96^2$)** | 10 | 6 | 2 | 2 |
| **Large Boxes ($\ge 96^2$)** | 13 | 9 | 1 | 3 |

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
