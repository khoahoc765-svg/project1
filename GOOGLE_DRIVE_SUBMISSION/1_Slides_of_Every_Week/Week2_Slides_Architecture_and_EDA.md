# Week 2 Presentation: Architecture Selection (YOLOv11n) & EDA
**Project:** Automated Food & Produce Freshness Detection System using YOLOv11  
**Student:** Trần Lê Xuân Mai (`24020011`)

---

## Slide 1: What, When, Why, Who, How Much (Week 2)
- **WHAT:** Model Architecture Selection (`YOLOv11n`), Exploratory Data Analysis (EDA), and 4 Core Data Risk Definitions.
- **WHEN:** Week 2 Milestone (Cumulative: 33h planned / 34h spent).
- **WHY:** `YOLOv11n` (`2.59M` parameters, `6.4 GFLOPs`) provides the optimal trade-off between high detection accuracy (`mAP`) and real-time CPU/Edge inference speed.
- **WHO:** Trần Lê Xuân Mai (`24020011`) — Architecture Benchmarking & Dataset Risk Analysis.
- **HOW MUCH:** Planned: `18 hours` | Spent: `19 hours` | Gap: `+1 hour` (spent inspecting raw Roboflow label quality).

---

## Slide 2: Why YOLOv11n? (Architecture Justification)
- **Backbone (`C3k2` + `SPPF` + `C2PSA`):** Spatial attention (`C2PSA`) focuses on small localized rotten spots and mold textures.
- **Neck (`PANet`):** Multi-scale feature fusion handles fruits at varying camera distances.
- **Real-Time Capability:** Processes static images and real-time video streams at `>30 FPS` on CPU and `~250 FPS` on T4 GPU.

---

## Slide 3: Top 4 Data Quality Risks Identified in Week 2
1. **Class Symmetry & Balance:** Ensuring equal representation of `Fresh` and `Rotten` samples to minimize False Negatives.
2. **Data Leakage Prevention:** Preserving strict partition isolation across `train`, `valid`, and `test`.
3. **HSV Color Distortion Control:** Preventing aggressive Hue shifts from turning fresh fruit colors into rotten brown tones.
4. **Lab Bias & Occlusion:** Overcoming single-item studio backgrounds using composite Mosaic augmentation.
