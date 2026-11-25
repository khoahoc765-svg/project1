# Automated Food and Produce Freshness Detection System for FIFO Inventory Optimization Using Controlled-Augmentation YOLOv11n

**Trần Lê Xuân Mai**  
*Faculty of Information and Communication Technology (ICT)*  
*VN-UK Institute for Research and Executive Education (VNUK), The University of Danang*  
Da Nang, Vietnam  
Student ID: `24020011` | Email: `24020011@vnuk.edu.vn`

---

## Abstract
Manual quality inspection of perishable fruits and vegetables in supermarkets and cold-chain warehouses is labor-intensive, subjective, and prone to false-negative sorting errors, impeding strict First-In-First-Out (FIFO) inventory rotation. This paper presents an end-to-end, real-time object detection and freshness grading system built upon the ultra-lightweight **YOLOv11n** architecture (`2.59M` parameters, `6.4 GFLOPs`). To ensure robust generalization without corrupting critical visual spoilage cues, we introduce a **5-stage Data Quality Audit** featuring cryptographic MD5 anti-leakage verification across dataset splits and a **Controlled HSV & Mosaic Augmentation** regimen (`hsv_h = 0.005`, `hsv_s = 0.3`, `hsv_v = 0.3`, `mosaic = 1.0`). Focusing on three representative produce archetypes across six classes (*Fresh/Rotten Apple, Banana, and Bell Pepper*), our fine-tuned YOLOv11n model trained with the `AdamW` optimizer and Cosine Annealing learning rate schedule achieves an overall **Precision of 98.6%**, **Recall of 100.0%**, **mAP@50 of 99.5%**, and **mAP@50-95 of 96.7%** on the independent test partition while operating at **27.4 ms per frame (~36 FPS) on a standard CPU**. The trained checkpoint (`5.21 MB`) is deployed in a modular local execution suite encompassing batch CLI inference and a live webcam web dashboard with real-time HSV and surface dark-spot analytics.

**Index Terms**—Object Detection, YOLOv11, Food Freshness Grading, Controlled Data Augmentation, Computer Vision, Smart Warehousing, FIFO Inventory Optimization.

---

## I. Introduction

Post-harvest loss in the retail and quick-commerce food supply chain accounts for significant economic waste globally. In modern distribution centers, produce sorting still relies heavily on human inspectors who visually evaluate surface color, bruising, and fungal decay. Under high-throughput conveyor operations, manual inspection suffers from visual fatigue and inconsistent grading standards, often allowing decayed items (*False Negatives*) to contaminate adjacent fresh stock and disrupting First-In-First-Out (FIFO) inventory management.

Recent advances in single-stage deep learning object detectors offer a compelling pathway toward automated, non-destructive optical sorting. However, deploying deep neural networks for produce freshness detection introduces four domain-specific challenges often overlooked in generic object detection pipelines:
1. **Color Sensitivity to Data Augmentation:** Unlike rigid objects (e.g., vehicles) that remain invariant under large hue perturbations, fruit freshness is directly encoded in surface chromaticity. Aggressive Hue (`hsv_h`) augmentation artificially shifts fresh red or yellow skin into brown rotten hues, poisoning supervision signals.
2. **Laboratory Bias and Occlusion:** Public datasets frequently contain isolated studio photographs of single fruits, causing models to fail when multiple items overlap on industrial conveyor belts.
3. **Annotation Noise and Data Leakage:** Pre-augmented community datasets often suffer from empty bounding-box label files and duplicate images leaked across training and evaluation partitions.
4. **Edge Hardware Constraints:** Warehouse sorting nodes require compact model weights (`< 10 MB`) capable of real-time inference on standard CPUs or embedded edge devices without costly server GPUs.

In this work, we design, audit, train, and deploy a complete end-to-end solution based on **Ultralytics YOLOv11n**. Our primary contributions are:
- **Strategic Proof-of-Concept (PoC) Taxonomy:** A balanced 6-class taxonomy covering three distinct visual surface archetypes: *Apple* (glossy spherical surface), *Banana* (elongated curvature with melanin dark-spot decay), and *Bell Pepper* (multi-lobed geometry with a calyx/stem).
- **Cryptographic Data Quality Audit:** An automated preprocessing audit (`step2_quality_audit.py`) that purges corrupt `.txt` annotations and verifies zero cross-split image leakage via MD5 hashing.
- **Controlled HSV & Mosaic Training Regimen:** A domain-tailored augmentation strategy that locks Hue perturbation (`hsv_h = 0.005`) while leveraging illumination variance (`hsv_v = 0.3`) and 4-image composite Mosaic training (`mosaic = 1.0`, `close_mosaic = 5`).
- **End-to-End Local Deployment:** A reproducible software suite (`train.py`, `detect.py`, `test_model.py`, `web_app.py`) achieving **99.5% mAP@50** and **100.0% Recall** on unseen test data.

---

## II. Dataset Survey and Automated Quality Audit

### A. Dataset Acquisition and Strategic Scope
We source our base annotations from the Roboflow Freshness Detection repository (`freshness-detection-vwn1a/freshness-detection-rhrze`), which aggregates over 21,000 raw images. Rather than training on dozens of severely imbalanced fruit categories—which degrades mean Average Precision (mAP) and causes memory bottlenecks on free cloud GPUs—we curate a balanced 6-class benchmark across three produce groups:
$$\mathcal{C} = \{\text{Fresh Apple}, \text{Fresh Banana}, \text{Fresh Bell Pepper}, \text{Rotten Apple}, \text{Rotten Banana}, \text{Rotten Bell Pepper}\}$$

### B. Five-Criteria Data Quality Audit
Before training, our automated audit pipeline (`step2_quality_audit.py`) enforces five strict quality gates, logging results to `reports/audit.json`:
1. **Annotation Validity:** Automatically scans all YOLO `.txt` label files, removing empty files and bounding boxes whose normalized coordinates $(x_c, y_c, w, h)$ fall outside $[0, 1]$.
2. **Class Symmetry and Balance:** Across 174 validated local benchmark images (`120 train`, `30 valid`, `24 test`) containing 367 bounding boxes, the distribution between fresh and rotten instances is verified at **185 Fresh (50.4%)** versus **182 Rotten (49.6%)** (`ratio = 50.3% : 49.7%`).
3. **MD5 Anti-Leakage Verification:** Every image file $I_k$ is hashed using 128-bit MD5:
   $$h(I_k) = \text{MD5}(I_k)$$
   Pairwise intersections across partitions confirm zero overlap:
   $$|\mathcal{H}_{\text{train}} \cap \mathcal{H}_{\text{valid}}| = |\mathcal{H}_{\text{train}} \cap \mathcal{H}_{\text{test}}| = |\mathcal{H}_{\text{valid}} \cap \mathcal{H}_{\text{test}}| = 0$$
4. **Chromatic Integrity Preservation:** Standard YOLO training uses `hsv_h = 0.015`. We restrict Hue variation to `hsv_h = 0.005` ($\pm 0.9^\circ$ on the color wheel) to preserve true necrotic browning and fungal mold colors.
5. **Lab Bias Mitigation:** We set `mosaic = 1.0` to synthesize 4-image composites during training, simulating clustered and partially occluded produce on conveyor belts.

---

## III. Methodology and YOLOv11n Architecture

### A. YOLOv11n Network Architecture
We select the nano variant **YOLOv11n** (`101 fused layers`, `2,598,266 parameters`, `6.4 GFLOPs`), structured into three modules:
1. **Backbone (`C3k2` + `SPPF` + `C2PSA`):** Replaces traditional `C2f` blocks with lightweight `C3k2` cross-stage partial blocks and appends a Cross-Stage Partial with Spatial Attention (`C2PSA`) module after Spatial Pyramid Pooling-Fast (`SPPF`). The `C2PSA` attention mechanism enhances sensitivity to localized surface defects (e.g., 5–10 pixel dark spots on rotten bananas).
2. **Neck (`PANet`):** Path Aggregation Network fusing multi-scale feature maps across strides $\{8, 16, 32\}$.
3. **Decoupled Detection Head:** Separates classification logits (Binary Cross-Entropy Loss $\mathcal{L}_{\text{cls}}$) from bounding-box regression (Complete IoU Loss $\mathcal{L}_{\text{CIoU}}$ and Distribution Focal Loss $\mathcal{L}_{\text{DFL}}$):
   $$\mathcal{L}_{\text{total}} = \lambda_{\text{box}}\mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}}\mathcal{L}_{\text{DFL}}$$
   where $\lambda_{\text{box}} = 7.5$, $\lambda_{\text{cls}} = 0.5$, and $\lambda_{\text{dfl}} = 1.5$.

### B. Transfer Learning and Optimization Protocol
We initialize the backbone from official COCO pretrained weights (`yolo11n.pt`) and fine-tune using the decoupled weight decay optimizer **AdamW** (`lr0 = 0.001`, `lrf = 0.01`, `weight_decay = 0.0005`, `warmup_epochs = 2.0`) governed by a **Cosine Annealing** schedule over $T = 20$ epochs:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$
Mosaic augmentation is disabled during the final 5 epochs (`close_mosaic = 5`) to refine bounding-box boundary precision on uncropped images.

---

## IV. Experimental Results and Quantitative Analysis

### A. Training Convergence Progression
Table I summarizes the training loss decay and validation metric convergence recorded in `runs/detect/train/results.csv`.

**TABLE I: Training and Validation Convergence Across Epochs (`YOLOv11n`)**

| Epoch | Train Box Loss | Train Cls Loss | Train DFL Loss | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.3802 | 3.3106 | 1.3204 | 62.66% | 34.48% | 62.62% | 48.03% |
| **5** | 0.6746 | 1.3095 | 0.9901 | 90.02% | 94.37% | 96.79% | 81.91% |
| **10** | 0.5621 | 0.8712 | 0.9317 | 97.69% | 98.48% | 99.50% | 90.09% |
| **15** | 0.4910 | 0.6694 | 0.9163 | 98.60% | 100.00% | 99.50% | 93.36% |
| **20** | **0.3256** | **0.3820** | **0.8354** | **98.90%** | **100.00%** | **99.50%** | **96.51%** |

### B. Independent Test Set Performance
We evaluate the best checkpoint (`runs/detect/train/weights/best.pt`, `5.21 MB`) on the unseen test split (`24 images`, `47 ground-truth instances`). Table II presents the per-class breakdown logged in `reports/evaluation_metrics.json`.

**TABLE II: Per-Class Quantitative Evaluation on Unseen Test Split**

| Class Name | Test Images | Ground-Truth Instances | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50-95 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes (Mean)** | **24** | **47** | **98.58%** | **100.00%** | **99.50%** | **96.69%** |
| `Fresh Apple` | 9 | 9 | 98.33% | 100.00% | 99.50% | 99.50% |
| `Fresh Banana` | 7 | 7 | 100.00% | 100.00% | 99.50% | 92.41% |
| `Fresh Bell Pepper` | 9 | 9 | 98.20% | 100.00% | 99.50% | 99.50% |
| `Rotten Apple` | 7 | 7 | 97.96% | 100.00% | 99.50% | 99.50% |
| `Rotten Banana` | 7 | 7 | 98.92% | 100.00% | 99.50% | 89.70% |
| `Rotten Bell Pepper` | 8 | 8 | 97.90% | 100.00% | 99.50% | 99.50% |

### C. Confusion Matrix and Runtime Analysis
1. **Zero False Negatives on Spoiled Produce:** As verified by `reports/confusion_matrix.png`, Recall across all three `Rotten` categories reaches **100.00%**, ensuring no decayed produce escapes detection on the test set.
2. **Real-Time CPU Inference Speed:** On an Intel Core i5-8257U CPU (`1.40 GHz`) without GPU acceleration, average inference latency is **27.4 ms per image** (`0.8 ms` preprocess, `27.4 ms` inference, `0.7 ms` postprocess), equivalent to **~36 FPS**.

---

## V. System Deployment, Limitations, and Future Work

### A. Dual Local Deployment Suite
The trained model is integrated into two production-ready local interfaces:
1. **Batch CLI Evaluator (`detect.py` & `test_model.py`):** Processes image directories and outputs color-coded bounding boxes (**Green** for `Fresh`, **Red** for `Rotten`).
2. **Interactive Web & Live Webcam Dashboard (`web_app.py`):** Serves a real-time inspection UI at `http://localhost:8085`, computing auxiliary HSV statistics $(\bar{H}, \bar{S}, \bar{V})$ and surface dark-spot ratio ($\% \text{ pixels with } V < 75$) within each detected bounding box.

### B. Limitations and Future Engineering Roadmap
1. **2D Single-View Blindspot $\rightarrow$ 360° Roller Conveyors:** A single top-down camera cannot observe bruises resting against the conveyor surface. Future physical integration will pair synchronized dual cameras with rotating rollers.
2. **Internal Core Spoilage $\rightarrow$ NIR / Hyperspectral Fusion:** Standard RGB sensors only inspect surface appearance. Integrating Near-Infrared (NIR, 700–1100 nm) imaging will enable non-destructive detection of internal tissue breakdown.
3. **Edge AI & 3-Tier FIFO ERP Integration:** Exporting the `5.21 MB` checkpoint to TensorRT on an NVIDIA Jetson Nano and combining YOLO confidence with our HSV dark-spot ratio will enable 3-tier automated warehouse routing: **Grade A** (*Long-term cold storage*), **Grade B** (*Immediate FIFO dispatch*), and **Grade C** (*Automated rejection*).

---

## VI. Conclusion

We have presented a complete, reproducible deep learning solution for automated produce freshness detection using **YOLOv11n**. By combining cryptographic MD5 data auditing, color-preserving HSV augmentation (`hsv_h = 0.005`), Mosaic occlusion training (`mosaic = 1.0`), and `AdamW` Transfer Learning, our compact `5.21 MB` model achieves **99.5% mAP@50** and **100.0% Recall** at **~36 FPS on CPU**, providing an effective foundation for real-time FIFO inventory optimization.

---

## References
1. G. Jocher, J. Qiu, and A. Chaurasia, "Ultralytics YOLOv11: State-of-the-Art Real-Time Object Detection," GitHub Repository, 2024. [Online]. Available: `https://github.com/ultralytics/ultralytics`
2. Roboflow Universe, "Freshness Detection Dataset (`freshness-detection-rhrze`)," 2024. [Online]. Available: `https://universe.roboflow.com/freshness-detection-vwn1a/freshness-detection-rhrze`
3. A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection," *arXiv preprint arXiv:2004.10934*, 2020.
4. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.
5. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2014, pp. 740–755.
