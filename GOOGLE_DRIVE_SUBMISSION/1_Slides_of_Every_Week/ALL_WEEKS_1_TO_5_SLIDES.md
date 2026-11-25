# Weekly Progress Slides (Week 1 to Week 5)
**Project:** Automated Food & Produce Freshness Detection System using YOLOv11  
**Student:** Trần Lê Xuân Mai (`24020011`)  
**Course:** Artificial Intelligence 2026 – Group 3

---

# 📅 WEEK 1 SLIDES: Problem Definition & Dataset Survey

## Slide 1.1: Project Overview & Practical Problem
- **Problem Statement:** Manual fruit and vegetable sorting in supermarkets and cold-chain warehouses is slow, subjective, and error-prone, hindering **FIFO (First-In, First-Out)** inventory execution and causing high food waste.
- **AI Objective:** Build an end-to-end **Object Detection + Freshness Classification** pipeline capable of localizing produce on conveyor belts and grading freshness in real time.
- **Dual Task Formulation:**
  1. **Spatial Localization (`Bounding Box`):** Detect where each fruit/vegetable is located.
  2. **Freshness Grading (`Classification`):** Classify each item as `Fresh` (Store/Sell) or `Rotten` (Reject).

## Slide 1.2: Dataset Survey & Strategic PoC Scope
- **Data Source:** Roboflow Universe (`freshness-detection-vwn1a / freshness-detection-rhrze`, >21,000 raw annotated images).
- **Strategic Proof-of-Concept (PoC) Scope (6 Classes = 3 Produce Types × 2 States):**
  - `Fresh Apple` vs. `Rotten Apple` *(Glossy spherical surface)*
  - `Fresh Banana` vs. `Rotten Banana` *(Curved geometry, dark-spot pigment decay)*
  - `Fresh Bell Pepper` vs. `Rotten Bell Pepper` *(Multi-lobed structure with green stem)*
- **SWOT Analysis:**
  - **Strengths:** Large pre-annotated bounding box dataset.
  - **Weaknesses:** Raw dataset contains noisy over-augmented samples and empty `.txt` labels.
  - **Opportunities:** High real-world warehouse applicability; fast Transfer Learning with YOLOv11.
  - **Threats:** Warehouse lighting variations and overlapping produce occlusion.

---

# 📅 WEEK 2 SLIDES: Model Selection (YOLOv11n) & Data Quality Risks

## Slide 2.1: Model Selection Justification — Why YOLOv11n?
- **Selected Architecture:** **Ultralytics YOLOv11n (Nano)** (`101 fused layers`, `2.59M parameters`, `6.4 GFLOPs`).
- **Key Advantages:**
  1. **Real-Time Edge Inference:** Processes frames in `~27.4 ms` on CPU (~36 FPS) and `~3.9 ms` on T4 GPU (~250 FPS).
  2. **Enhanced Feature Extraction:** Incorporates `C3k2`, `SPPF`, and `C2PSA` (Spatial Attention) blocks to capture fine-grained surface bruises and mold spots.
  3. **Transfer Learning Efficiency:** Pretrained COCO weights (`yolo11n.pt`) enable rapid convergence.

## Slide 2.2: Exploratory Data Analysis (EDA) & 4 Critical Data Risks
1. **Risk 1 — Class Symmetry & Balance:** Imbalance between `Fresh` and `Rotten` classes leads to high False Negatives (missed rotten items).
2. **Risk 2 — Data Leakage Across Splits:** Re-splitting pre-augmented images leaks near-identical samples between `train` and `test`.
3. **Risk 3 — HSV Color Distortion:** Standard high Hue augmentation (`hsv_h`) shifts fresh red/yellow fruit into brown/rotten tones, corrupting ground-truth labels.
4. **Risk 4 — Lab Bias & Occlusion:** Single-fruit studio photos fail when fruits overlap on real conveyor belts.

---

# 📅 WEEK 3 SLIDES: Data Preprocessing Pipeline & Baseline Model

## Slide 3.1: Automated Data Quality Audit (`step2_quality_audit.py`)
| Audit Criterion | Current Status | Engineering Solution Implemented | Verified Result |
| :--- | :--- | :--- | :--- |
| **1. Label Validity** | Empty/corrupt `.txt` files found | Automated bounding-box coordinate & empty-file purging | **100% invalid labels removed** |
| **2. Class Balance** | Needs Fresh vs. Rotten symmetry | Automated box counter & `class_distribution.png` | **50.3% Fresh : 49.7% Rotten** |
| **3. Anti-Leakage** | High risk if splits overlap | **MD5 Cryptographic Hash Check** across `train/valid/test` | **0 duplicate images (`PASSED`)** |
| **4. Color Preservation** | Sensitive to Hue shifts | Locked **`hsv_h = 0.005`**, tuned **`hsv_s = 0.3, hsv_v = 0.3`** | Preserves true bruise/mold color |
| **5. Lab Bias Mitigation** | Isolated single-fruit shots | Enabled **`mosaic = 1.0`** (4-image composite training) | Handles overlapping produce |

## Slide 3.2: Week 3 Baseline Model Results (15 Epochs)
- **Baseline Setup:** `YOLOv11n`, 15 epochs, default optimizer settings.
- **Baseline Metrics:**
  - Precision: `81.2%` | Recall: `77.5%` | mAP@50: `83.4%` | mAP@50-95: `61.8%`
- **Week 4 Action Plan:** Upgrade to `AdamW` with Cosine Annealing LR, apply Controlled HSV + Mosaic Augmentation, and migrate to a modular local **VS Code** environment.

---

# 📅 WEEK 4 SLIDES: Hyperparameter Tuning, VS Code Migration & Evaluation

## Slide 4.1: Controlled Augmentation & Transfer Learning (`train.py`)
- **Hybrid Workflow:** Migrated from monolithic Colab notebooks to a modular local **VS Code (`.venv`)** project (`config.py`, `train.py`, `detect.py`, `test_model.py`, `web_app.py`).
- **Tuned Hyperparameters:**
  - Pretrained Backbone: `yolo11n.pt` (Transfer Learning)
  - Optimizer: `AdamW` (`lr0 = 0.001`, `lrf = 0.01`, `cos_lr = True`, `warmup_epochs = 2.0`, `seed = 0`)
  - Controlled Augmentation: `mosaic = 1.0` (`close_mosaic = 5`), `hsv_h = 0.005`, `hsv_s = 0.3`, `hsv_v = 0.3`

## Slide 4.2: Quantitative Performance Jump & Confusion Matrix Analysis
- **Validation & Unseen Test Set Evaluation (`24 test images, 47 produce instances`):**
  - **Precision:** `98.6%` *(up from 81.2% baseline)*
  - **Recall:** `100.0%` *(zero missed rotten items on test set)*
  - **mAP@50:** `99.5%` *(up from 83.4% baseline)*
  - **mAP@50-95:** `96.7%` *(tight bounding-box localization)*

---

# 📅 WEEK 5 SLIDES: Final Integration, Critical Reflection & GitHub Review

## Slide 5.1: Full Deep Learning Pipeline (`Data -> Model -> Output`)
1. **Stage 1 (Data):** Roboflow ingestion (`step1_prepare_data.py`) $\rightarrow$ MD5 Zero-Leakage & Empty Label Audit (`step2_quality_audit.py`).
2. **Stage 2 (Model):** YOLOv11n Transfer Learning with Controlled HSV/Mosaic (`train.py`) $\rightarrow$ Compact `5.21 MB best.pt`.
3. **Stage 3 (Output):**
   - **CLI Batch Evaluator (`test_model.py`, `detect.py`):** Color-coded bounding boxes (**Green = Fresh**, **Red = Rotten**) + HSV & dark-spot ratio analytics.
   - **Interactive Web & Live Webcam UI (`web_app.py`):** Real-time browser dashboard at `http://localhost:8085`.

## Slide 5.2: Limitations, Future Roadmap & 100% Reproducibility
- **Current Limitations:**
  1. *2D Single-View Blindspot:* Cannot inspect bruises hidden on the bottom side of fruit resting on the belt.
  2. *Internal Rot:* Subsurface core decay invisible to standard RGB cameras.
  3. *Extreme Specular Glare:* Strong reflections on glossy skin affect the HSV Value (`V`) channel.
- **Future Engineering Improvements:**
  1. *360° Multi-Camera Conveyor Rollers* for full-surface scanning.
  2. *NIR / Hyperspectral Sensors* to detect internal spoilage beneath the peel.
  3. *Edge TensorRT Deployment* on NVIDIA Jetson Nano with **3-Tier FIFO ERP Grading** (*Grade A: Store* $\rightarrow$ *Grade B: Export FIFO Now* $\rightarrow$ *Grade C: Reject*).
- **GitHub Reproducibility:** Modular codebase, `requirements.txt`, fixed `seed=0`, automated 1-click runner `run_pipeline.py`, and IEEE Final Report.
