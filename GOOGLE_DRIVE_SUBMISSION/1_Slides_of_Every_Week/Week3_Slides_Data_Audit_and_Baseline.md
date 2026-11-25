# Week 3 Presentation: Data Quality Audit & Baseline Model
**Project:** Automated Food & Produce Freshness Detection System using YOLOv11  
**Student:** Trần Lê Xuân Mai (`24020011`)

---

## Slide 1: What, When, Why, Who, How Much (Week 3)
- **WHAT:** Automated Data Quality Audit Script (`step2_quality_audit.py`), Controlled Preprocessing Pipeline, and 15-Epoch Baseline `YOLOv11n` Model.
- **WHEN:** Week 3 Milestone (Cumulative: 55h planned / 61h spent).
- **WHY:** Cleaning corrupt `.txt` labels and verifying zero data leakage via MD5 hashing guarantees trustworthy evaluation metrics before hyperparameter tuning.
- **WHO:** Trần Lê Xuân Mai (`24020011`) — Data Quality Engineering & Baseline Training.
- **HOW MUCH:** Planned: `22 hours` | Spent: `27 hours` | Gap: `+5 hours` (resolved by automating label cleaning and MD5 verification in Python).

---

## Slide 2: 5-Point Data Quality Audit Matrix (`reports/audit.json`)
| Criterion | Issue Addressed | Solution Implemented | Outcome |
| :--- | :--- | :--- | :--- |
| **1. Validity** | Empty or out-of-bounds `.txt` labels | Automated coordinate & empty-file scanner | **100% clean labels** |
| **2. Balance** | Fresh vs. Rotten class skew | Bounding-box distribution audit (`class_distribution.png`) | **50.3% Fresh : 49.7% Rotten** |
| **3. Anti-Leakage** | Duplicate images across splits | **MD5 Hash verification** across `train/valid/test` | **0 duplicates (`PASSED`)** |
| **4. Color Integrity** | Hue augmentation corrupting rot colors | Locked **`hsv_h = 0.005`**, tuned **`hsv_s = 0.3, hsv_v = 0.3`** | Preserves true bruise hues |
| **5. Lab Bias** | Isolated studio backgrounds | Enabled **`mosaic = 1.0`** | Simulates overlapping produce |

---

## Slide 3: Week 3 Baseline Results & Week 4 Tuning Plan
- **Baseline Performance (15 Epochs, Default Settings):**
  - Precision: `81.2%` | Recall: `77.5%` | mAP@50: `83.4%` | mAP@50-95: `61.8%`
- **Planned Tasks for Week 4:**
  1. Fine-tune hyperparameters with `AdamW` optimizer and Cosine Annealing learning rate.
  2. Analyze the 6-class **Confusion Matrix** to eliminate False Negatives on rotten produce.
  3. Migrate from Colab to a modular local **VS Code** project and build inference scripts (`detect.py`, `test_model.py`).
