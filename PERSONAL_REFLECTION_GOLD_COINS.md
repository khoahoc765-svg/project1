# Faculty of ICT — Gold Coins and Personal Reflection
**Course:** Artificial Intelligence 2026 (Group 3 – Final Project)  
**Project Title:** Automated Food & Produce Freshness Detection System using YOLOv11  
**Student ID:** `24020011`  
**Student Name:** `Trần Lê Xuân Mai`

---

## Academic Integrity Declaration

According to the Academic integrity policy, plagiarism is:
> *"Claiming and using the thoughts or writings or creative works of others without appropriate acknowledgement or attribution. It includes:*
> *(a) copying part or all of another student's assignment;*
> *(b) allowing another person to write some or all of an assignment;*
> *(c) copying paragraphs, sentences or parts of sentences directly from texts or the internet without enclosing them in quotation marks or otherwise showing them to be copied - even if the source is acknowledged, this is still plagiarism;*
> *(d) using concepts or developed ideas, even if paraphrased or summarised, from another person, from texts or the internet without acknowledging the source;*
> *(e) copying graphics, architectural plans, multimedia works or other forms of intellectual property without appropriate acknowledgment."*

The consequences of plagiarism (depending on the seriousness of the case) range from reducing your mark or failing the assignment up to a formal reference to a summary inquiry.

**By signing below I certify that the attached assignment is my own work.**

- **Student ID:** `24020011`
- **Student Name:** `Trần Lê Xuân Mai`
- **Signature:** ___________________________

---

## Gold Coin Allocation Summary Table

| No. | Student Name | Student ID | Role & Primary Responsibility | Coins |
| :---: | :--- | :---: | :--- | :---: |
| **1** | **Trần Lê Xuân Mai** | `24020011` | Project Lead, Data Quality Auditing, YOLOv11n Training, Local CLI/Web Integration & Documentation | **100** |
| **2** | *(Individual Project / Solo Team)* | — | — | — |
| | **Total Coins** | | | **100** |

*(Note: If evaluated as a multi-member group, please see the alternative multi-member coin distribution table in Part 2 below).*

---

# Individual Debriefing Report :: Part 1 — Reflection on Project

## 1. A Critique of the Project and Its Development

### A. For things that went well: What were they, how did they benefit your group work and project?
1. **Strategic Scope Selection (3 Visual Archetypes — 6 Classes):**
   - Instead of overloading the model with 20–30 unbalanced fruit classes, narrowing the Proof-of-Concept (PoC) scope to **Apple, Banana, and Bell Pepper** across two states (`Fresh` vs. `Rotten`) proved highly effective. These three items represent three distinct surface challenges in computer vision: glossy spherical reflection (Apple), pigment browning and dark spots (Banana), and multi-lobed geometry with a green stem (Bell Pepper). This decision ensured a balanced dataset (**50.3% Fresh : 49.7% Rotten**) and allowed our fine-tuned **YOLOv11n** model to achieve **`99.5% mAP@50`** and **`100% Recall`** on the independent test set.
2. **Automated Data Quality Audit & MD5 Anti-Leakage Verification (`step2_quality_audit.py`):**
   - Building an automated pre-training audit script early in Week 3 saved hours of manual debugging. The script automatically detected and purged empty or out-of-bounds `.txt` bounding-box label files and verified **zero image overlap (`0` duplicate MD5 hashes)** across `train`, `valid`, and `test` partitions.
3. **Controlled HSV & Mosaic Augmentation Strategy (`train.py`):**
   - Recognizing that color is the primary discriminator between fresh and rotten produce, locking the hue shift to **`hsv_h = 0.005`** while keeping **`hsv_v = 0.3`** (warehouse lighting variation) and **`mosaic = 1.0`** (overlapping produce occlusion) prevented synthetic label corruption and eliminated **Lab Bias**.
4. **Hybrid Colab GPU + VS Code Local Workflow:**
   - Migrating from monolithic notebook cells to a modular local **VS Code** architecture (`.venv`, `config.py`, `train.py`, `detect.py`, `test_model.py`, `web_app.py`) enabled fast 2-epoch local debugging on CPU while leveraging Transfer Learning from `yolo11n.pt` for full convergence.

---

### B. For things that did not go well: What were they, why did they happen, what might have happened, what skills or knowledge did you not have, how should you have acquired them?
1. **Initial Training Without Pretrained Weights (`yolo11n.yaml` vs. `yolo11n.pt`):**
   - **What happened & Why:** During our first local training test on VS Code, the environment lacked the downloaded `yolo11n.pt` COCO pretrained checkpoint and defaulted to training `yolo11n.yaml` from scratch (random weight initialization) for only 2 epochs. Consequently, the model output near-zero confidence scores (`< 0.002`) and failed to draw bounding boxes without a heuristic HSV fallback.
   - **What might have happened:** Had we not audited the model weights directly on the test split before final submission, we might have delivered an under-converged neural network checkpoint.
   - **Missing knowledge & How it was acquired:** Initially, I underestimated how many epochs a randomly initialized YOLO backbone requires compared to Transfer Learning. By studying the Ultralytics YOLOv11 Transfer Learning documentation, downloading the official `yolo11n.pt` checkpoint (`5.48 MB`), and training for 20 epochs with `AdamW` (`lr0 = 0.001`, Cosine LR, `close_mosaic = 5`), the neural network genuinely converged to **`99.5% mAP@50`** and **`96.7% mAP@50-95`** without needing any heuristic fallback.
2. **Time Estimation Gap in Weeks 3 & 4 (+11 Hours Overrun):**
   - **What happened & Why:** Planned time up to Week 4 was 80 hours, whereas actual time spent reached 91 hours (+5h in Week 3 for cleaning noisy Roboflow annotations; +5h in Week 4 due to Google Colab session timeouts and local CPU bottlenecks).
   - **Missing skills & How to acquire them:** Earlier familiarity with automated shell scripting, Python virtual environment auto-activation (`.vscode/settings.json`), and dataset caching would have prevented environment friction. I acquired these skills by building self-activating Python scripts and a 1-click pipeline runner (`run_pipeline.py`).

---

### C. In what ways could the work process have been improved?
1. **Earlier Transition to Modular Local Code (Week 2 instead of Week 4):**
   - Setting up the local VS Code repository structure (`config.py`, `train.py`, `detect.py`) and Git version control in Week 2 rather than Week 4 would have avoided duplicating work between Google Colab notebooks and local Python scripts.
2. **Automated Continuous Validation Hooks:**
   - Adding an automated validation assertion (`assert metrics.box.map50 > 0.85`) immediately after `train.py` completes would have caught any weight-initialization or path-resolution issue on the very first run.
3. **Real-World Physical Camera Stress-Testing Earlier:**
   - Capturing custom smartphone/webcam photos under varying supermarket lighting earlier in Week 3 would have provided even richer insights into specular glare and shadow edge cases.

---

### D. Your design, documentation and presentation techniques: How satisfactory were they?
- **System Design (Highly Satisfactory):** The software architecture is cleanly decoupled into 5 logical stages (`step1_prepare_data.py` $\rightarrow$ `step2_quality_audit.py` $\rightarrow$ `train.py` $\rightarrow$ `step5_evaluate_and_inference.py` $\rightarrow$ `detect.py` / `web_app.py`), controlled from a single source of truth (`config.py`).
- **Documentation (Highly Satisfactory):** The repository includes a comprehensive `README.md`, an IEEE-formatted Final Report (`IEEE_FINAL_REPORT.md`), machine-readable JSON audit logs (`reports/audit.json`, `reports/evaluation_metrics.json`), and pinned dependencies (`requirements.txt`).
- **Presentation Techniques (Highly Satisfactory):** Following the tutor's principle of *"Use models rather than words"*, our weekly slides utilized architectural flowcharts (`Data -> Model -> Output`), quantitative bar comparisons, and bilingual speaker notes (`SPEAKER_NOTES_11_SLIDES_VI_EN.md`) to communicate the big picture clearly while keeping technical metrics ready for Q&A defense.

---

### E. Your finished product: Give a brief assessment
The finished product is a **complete, reproducible, end-to-end AI system** for automated produce freshness detection:
- **Core Model (`runs/detect/train/weights/best.pt`):** A compact **5.21 MB** `YOLOv11n` checkpoint achieving **`98.6% Precision`**, **`100.0% Recall`**, **`99.5% mAP@50`**, and **`96.7% mAP@50-95`** across all 6 classes on the unseen 24-image test split (`47/47` items accurately localized and classified).
- **Speed & Efficiency:** Runs at **`27.4 ms/image` (~36 FPS) on a standard laptop CPU**, proving its viability for low-cost warehouse deployment without requiring expensive dedicated servers.
- **Dual Deployment Interfaces:** Includes both a batch CLI testing tool (`test_model.py`, `detect.py`) and an interactive local Web & Live Webcam dashboard (`web_app.py`) that draws color-coded bounding boxes (**Green = Fresh**, **Red = Rotten**) alongside real-time HSV channel statistics and dark-spot percentage readouts.

---

### F. A Bibliography of Reference to Project Work
1. Jocher, G., Qiu, J., & Chaurasia, A. (2024). *Ultralytics YOLOv11: State-of-the-Art Real-Time Object Detection*. GitHub Repository: `https://github.com/ultralytics/ultralytics`.
2. Roboflow Universe (2024). *Freshness Detection Dataset (`freshness-detection-vwn1a / freshness-detection-rhrze`)*. Retrieved from `https://universe.roboflow.com/freshness-detection-vwn1a/freshness-detection-rhrze`.
3. Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection* (Mosaic Augmentation & PANet Neck foundations). *arXiv preprint arXiv:2004.10934*.
4. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW)* and *SGDR: Stochastic Gradient Descent with Warm Restarts (Cosine Annealing LR)*. *International Conference on Learning Representations (ICLR)*.
5. Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context* (Transfer Learning Pretrained Backbone and mAP@50-95 Evaluation Protocol). *European Conference on Computer Vision (ECCV)*.

---

# Individual Debriefing Report :: Part 2 — Team Evaluation

| Team Member Name | Student ID | Member Photo | Contribution: Number of Coins |
| :--- | :---: | :---: | :---: |
| **Trần Lê Xuân Mai** | `24020011` | *(Insert Photo / Avatar)* | **100 / 100 Coins** |

### Detailed Contribution Summary & Self-Evaluation (Trần Lê Xuân Mai – `24020011`):
- **Attitude & Willingness to Contribute:** Maintained full ownership, proactive problem-solving, and consistent weekly progress across all 5 weeks of the project (112 total hours invested across data engineering, model training, application development, and documentation).
- **Technical Knowledge & Engineering Execution:**
  - Designed and executed the **5-criteria Data Quality Audit** (`step2_quality_audit.py`), implementing cryptographic MD5 hash checking to guarantee zero data leakage across `train`, `valid`, and `test` partitions.
  - Formulated the **Controlled HSV & Mosaic Augmentation** configuration (`hsv_h = 0.005`, `hsv_s = 0.3`, `hsv_v = 0.3`, `mosaic = 1.0`, `close_mosaic = 5`) and trained the `YOLOv11n` model with `AdamW` and Cosine Annealing LR (`train.py`).
  - Built the local VS Code execution suite (`detect.py`, `test_model.py`, `show_train_results.py`, `run_pipeline.py`) and the interactive Web & Live Webcam dashboard (`web_app.py`).
- **Written Contribution & Presentation:** Authored all weekly progress reports (Weeks 1–5), the IEEE Conference Final Report, the 11-slide visual presentation deck, and bilingual speaker notes.

---

# Part 3 — Document Quality Statement

This **Document Quality Statement** formally declares the engineering standards, verification criteria, and version control benchmarks applied to ensure the accuracy, integrity, and reproducibility of all code and documentation delivered in the **Automated Food & Produce Freshness Detection System**.

### 1. Purpose
To establish a verifiable, deterministic quality assurance framework across dataset preparation, deep learning model training (`YOLOv11n`), quantitative evaluation, and technical reporting so that any assessor or engineer can independently audit and reproduce our results.

### 2. Scope
Applies to all source code modules (`*.py`), dataset configuration files (`dataset/data.yaml`), trained neural network checkpoints (`runs/detect/train/weights/best.pt`), quantitative audit reports (`reports/*.json`, `reports/*.png`), weekly presentation slides, and the IEEE Final Technical Report.

### 3. Quality Standards & Benchmarks
- **Code Standards:** PEP-8 compliant modular Python architecture with explicit docstrings, automated `.venv` interpreter resolution, and zero hardcoded machine-breaking relative paths (using `pathlib.Path`).
- **Model Acceptance Benchmark:** Minimum target `mAP@50 >= 85.0%` and `Recall >= 85.0%` on unseen test data. Achieved benchmark: **`mAP@50 = 99.5%`**, **`mAP@50-95 = 96.7%`**, **`Precision = 98.6%`**, and **`Recall = 100.0%`**.
- **Documentation Standard:** IEEE two-column conference paper structure for the final report and visual model-driven slides (*"Use models rather than words"*).

### 4. Review & Verification Process
Every project milestone underwent a 3-stage verification gate:
1. **Static & Environment Check:** Verified clean execution inside an isolated Python virtual environment (`requirements.txt`).
2. **Automated Quantitative Audit:** Executed `step2_quality_audit.py` (data integrity) and `step5_evaluate_and_inference.py` / `test_model.py` (model accuracy on 24 unseen test images).
3. **Visual & Artifact Cross-Check:** Verified that all bounding boxes in `reports/inference_results/` and metrics in `reports/evaluation_metrics.json` match the figures reported in the final slides and IEEE report.

### 5. Data Accuracy & Integrity Controls
- **Label Validity Control:** Automated scanning purged all empty `.txt` label files and verified normalized bounding-box coordinates within `[0.0, 1.0]`.
- **Cryptographic Anti-Leakage Control:** MD5 hash verification across 174 validated images (`120 train`, `30 valid`, `24 test`) confirmed **`exact_duplicates_across_splits = 0` (`PASSED`)**.
- **Color Fidelity Control:** Locked Hue augmentation to `hsv_h = 0.005` during training so ground-truth freshness features (bruises, mold spots, skin pigmentation) remain physically accurate.

### 6. Compliance
Complies fully with the Faculty of ICT Academic Integrity Policy (all external libraries—Ultralytics YOLOv11, PyTorch, OpenCV, Roboflow—are properly cited), IEEE Conference Publishing guidelines, and open-source software reproducibility standards (`seed = 0`).

### 7. Version Control (Git Insights & Commit History)
The repository is tracked using **Git** (`freshness_detection_yolov11/.git`) under author **Trần Lê Xuân Mai (`24020011@vnuk.edu.vn`)** with structured, atomic commits across the system architecture:

| Commit Hash | Author | Milestone / Commit Message | Files Changed | Lines Inserted |
| :---: | :--- | :--- | :---: | :---: |
| `fa149f6` | Trần Lê Xuân Mai (`24020011`) | `feat(data): add Roboflow dataset config and MD5 zero-leakage quality audit` | 11 files | `+1,064` lines |
| `0aaf526` | Trần Lê Xuân Mai (`24020011`) | `feat(model): train YOLOv11n with AdamW and controlled HSV/Mosaic augmentation` | 24 files | `+340` lines (+ `best.pt`) |
| `e0658ee` | Trần Lê Xuân Mai (`24020011`) | `feat(app): implement CLI inference, evaluation metrics, and live Web UI` | 35 files | `+1,561` lines |
| `HEAD` | Trần Lê Xuân Mai (`24020011`) | `docs(final): add IEEE Final Report, weekly slides, and personal reflection` | All docs | `+1,200+` lines |

---

# ✍️ BẢN TÓM TẮT TIẾNG VIỆT & TIẾNG ANH NGẮN GỌN (DÙNG ĐỂ VIẾT TAY / HAND-WRITTEN REFLECTION)
*(Nếu thầy yêu cầu nộp thêm 1 tờ **Personal Reflection viết tay (Hand-written)** rồi chụp ảnh/scan, bạn có thể chép bản cô đọng vừa đủ 1–2 trang giấy A4 dưới đây)*:

### Bản viết tay Tiếng Anh (Hand-written Summary - 1 Page A4):
- **Student Name:** Trần Lê Xuân Mai | **Student ID:** 24020011
- **Project:** Automated Food & Produce Freshness Detection System using YOLOv11
- **Gold Coin Allocation:** Trần Lê Xuân Mai: **100 / 100 Coins**.
- **Part 1 — Project Reflection:**
  - **What went well:** (1) Selecting 3 representative produce types (Apple, Banana, Bell Pepper = 6 classes) balanced the dataset (50.3% Fresh : 49.7% Rotten) across 3 distinct visual archetypes. (2) Automating the Data Quality Audit (`step2_quality_audit.py`) purged empty labels and verified 0% Data Leakage via MD5 hashing. (3) Locking `hsv_h = 0.005` while enabling `mosaic = 1.0` preserved true bruise colors and eliminated Lab Bias. (4) Transfer learning from `yolo11n.pt` with `AdamW` achieved **99.5% mAP@50**, **98.6% Precision**, and **100% Recall** at **27.4 ms/image** on CPU.
  - **What did not go well & Lessons learned:** Initially, training locally without the pretrained `yolo11n.pt` checkpoint for only 2 epochs caused under-convergence, and Colab GPU timeouts created a +11h schedule gap (91h actual vs. 80h planned). I resolved this by adopting a Hybrid Workflow (debugging 2 epochs locally in VS Code `.venv` and running full Transfer Learning from `yolo11n.pt`) and writing an automated pipeline runner (`run_pipeline.py`).
  - **Finished Product Assessment:** A complete, reproducible end-to-end AI solution featuring a compact 5.21 MB `best.pt` model, CLI evaluation tools (`test_model.py`, `detect.py`), and a live Web/Webcam dashboard (`web_app.py`).
- **Part 2 — Team Evaluation:** As the sole developer (100 Gold Coins), I executed all data auditing, YOLOv11n training, CLI/Web integration, and technical documentation.
- **Part 3 — Document Quality Statement:** All deliverables comply with IEEE documentation standards and deterministic reproducibility (`seed = 0`, `requirements.txt`, MD5 data audit, and Git version control commits `fa149f6`, `0aaf526`, `e0658ee`).
