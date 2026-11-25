# Faculty of ICT — Gold Coins and Personal Reflection

According to the Academic integrity policy, plagiarism is:
> *"Claiming and using the thoughts or writings or creative works of others without appropriate acknowledgement or attribution. It includes:*
> *(a) copying part or all of another student's assignment;*
> *(b) allowing another person to write some or all of an assignment;*
> *(c) copying paragraphs, sentences or parts of sentences directly from texts or the internet without enclosing them in quotation marks or otherwise showing them to be copied - even if the source is acknowledged, this is still plagiarism;*
> *(d) using concepts or developed ideas, even if paraphrased or summarised, from another person, from texts or the internet without acknowledging the source;*
> *(e) copying graphics, architectural plans, multimedia works or other forms of intellectual property without appropriate acknowledgment."*

The consequences of plagiarism (depending on the seriousness of the case) range from reducing your mark or failing the assignment up to a formal reference to a summary inquiry.

**By signing below I certify that the attached assignment is my own work.**

| Student Id | Student Name | Signature |
| :---: | :---: | :---: |
| **`24020011`** | **Trần Lê Xuân Mai** | *Trần Lê Xuân Mai* |

### Assessors Comments & Grade:
| Assessor Comments | Grade |
| :--- | :---: |
| *(For Lecturer / Tutor Use)* | |

### Gold Coin Allocation:

| No. | Student name | Coins |
| :---: | :--- | :---: |
| **1** | **Trần Lê Xuân Mai (`24020011` — Individual Project)** | **100** |
| **2** | — | — |
| **3** | — | — |
| **4** | — | — |
| **5** | — | — |
| | **Total Coins** | **100** |

---

# Individual Debriefing Report :: Part 1 — Reflection on Project

### 1. For things that went well — what were they, how did they benefit your group work and project?
* **Strategic Proof-of-Concept Scope (6 Classes across 3 Visual Archetypes):** Instead of training on 20–30 unbalanced fruit types which would dilute accuracy and exhaust GPU memory, focusing on **Apple** (glossy skin), **Banana** (dark-spot browning), and **Bell Pepper** (multi-lobed shape with green stem) ensured a balanced dataset (**50.3% Fresh : 49.7% Rotten**) and enabled our `YOLOv11n` model to reach **`99.5% mAP@50`**.
* **Automated Data Quality Audit & MD5 Anti-Leakage Check (`step2_quality_audit.py`):** Building an automated audit script early in Week 3 purged empty/corrupt `.txt` bounding-box labels and verified **zero duplicate MD5 hashes** across `train`, `valid`, and `test` splits, guaranteeing trustworthy evaluation.
* **Domain-Specific Controlled Augmentation (`hsv_h = 0.005`, `mosaic = 1.0`):** Locking Hue perturbation to `hsv_h = 0.005` preserved natural necrotic bruise and mold colors, while `mosaic = 1.0` overcame single-fruit studio **Lab Bias** and handled overlapping produce.
* **Modular Local VS Code Architecture:** Migrating from Google Colab notebooks to a clean local VS Code project (`.venv`, `config.py`, `train.py`, `detect.py`, `test_model.py`, `web_app.py`) made debugging, CLI batch testing, and live webcam demonstrations fast and reproducible.

---

### 2. For things that did not go well — what were they, why did they happen, what might have happened, what skills or knowledge did you not have, how should you have acquired them?
* **Initial Local Training Without Pretrained Weights (`yolo11n.yaml` vs. `yolo11n.pt`):**
  * *What happened & why:* When first testing locally in VS Code, the COCO pretrained `yolo11n.pt` checkpoint had not been downloaded yet, so the script trained random weights from scratch for only 2 debug epochs, producing near-zero confidence (`< 0.002`).
  * *What might have happened:* Had I not audited `best.pt` directly on the unseen test split, I might have submitted an under-converged model.
  * *Missing knowledge & how it was acquired:* I initially lacked practical knowledge of how strongly YOLO relies on Transfer Learning initialization for small epoch budgets. I resolved this by studying Ultralytics Transfer Learning docs, downloading `yolo11n.pt` (`5.48 MB`), and training for 20 epochs with `AdamW` and `close_mosaic = 5`, achieving **`99.5% mAP@50`** and **`100.0% Recall`**.
* **Schedule Overrun in Weeks 3–4 (+11 Hours Gap: 80h Planned vs. 91h Spent):**
  * *What happened & why:* Cleaning noisy Roboflow annotations (+5h in Week 3) and dealing with Colab GPU disconnects, slow local CPU runs, and relative path errors in `data.yaml` (+5h in Week 4) caused delays.
  * *Missing skills & how to acquire them:* I initially lacked experience with VS Code virtual environment configuration (`.vscode/settings.json`) and `pathlib` dynamic path resolution, which I should have practiced in Week 1 before scaling experiments.

---

### 3. In what ways could the group work / project execution have been improved?
* **Earlier Modularization & Version Control:** Initializing the Git repository and modular Python structure in Week 1 rather than Week 4 would have eliminated code duplication between Colab notebooks and VS Code scripts.
* **Automated Post-Training Validation Assertions:** Adding an automatic validation check (`mAP@50 > 0.85` immediately after `train.py`) would have caught missing pretrained weights immediately.
* **Earlier Peer Feedback & Real Lighting Stress-Tests:** Conducting earlier peer reviews with Group 3 classmates and capturing physical phone/webcam photos under harsh warehouse shadows in Week 3 would have surfaced glare and 2D blindspot limitations sooner.

---

### 4. Your design, documentation and presentation techniques — How satisfactory were they?
* **System Design (Highly Satisfactory):** Cleanly decoupled into 5 modular stages (`step1_prepare_data.py` $\rightarrow$ `step2_quality_audit.py` $\rightarrow$ `train.py` $\rightarrow$ `step5_evaluate_and_inference.py` $\rightarrow$ `detect.py` / `web_app.py`) governed by `config.py`.
* **Documentation (Highly Satisfactory):** Delivered a complete `README.md`, an IEEE two-column Final Report, JSON audit/evaluation logs (`reports/audit.json`, `reports/evaluation_metrics.json`), and `requirements.txt`.
* **Presentation Techniques (Highly Satisfactory):** Applied the tutor's *"Use models rather than words"* principle using end-to-end architecture diagrams (`Data -> Model -> Output`), metric comparison charts, and live VS Code / Web UI demos.

---

### 5. Your finished product — give a brief assessment.
The finished system is a complete, deterministic, end-to-end AI solution for produce freshness grading. Powered by a compact **5.21 MB `YOLOv11n` checkpoint (`best.pt`)**, it achieves **`98.6% Precision`**, **`100.0% Recall`** (zero missed rotten produce), **`99.5% mAP@50`**, and **`96.7% mAP@50-95`** across all 6 classes on the unseen 24-image test split (`47/47` items detected) at **`27.4 ms/frame` (~36 FPS)** on a standard laptop CPU. It provides both batch CLI tools (`test_model.py`, `detect.py`) and an interactive Web/Webcam dashboard (`web_app.py`) with real-time HSV and surface dark-spot analytics. While 2D single-camera blindspots and internal core rot remain physical limitations, the software fully achieves its Proof-of-Concept objectives.

---

### 6. A Bibliography of reference to group work
1. G. Jocher, J. Qiu, and A. Chaurasia, *"Ultralytics YOLOv11: State-of-the-Art Real-Time Object Detection,"* GitHub Repository, 2024. `https://github.com/ultralytics/ultralytics`
2. Roboflow Universe, *"Freshness Detection Dataset (`freshness-detection-rhrze`),"* 2024. `https://universe.roboflow.com/freshness-detection-vwn1a/freshness-detection-rhrze`
3. A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, *"YOLOv4: Optimal Speed and Accuracy of Object Detection"* (Mosaic Augmentation & PANet), *arXiv:2004.10934*, 2020.
4. I. Loshchilov and F. Hutter, *"Decoupled Weight Decay Regularization (AdamW),"* in *Proc. ICLR*, 2019.
5. T.-Y. Lin et al., *"Microsoft COCO: Common Objects in Context,"* in *Proc. ECCV*, 2014, pp. 740–755.

---

# Individual Debriefing Report :: Part 2 — Team Evaluation

| Team Member Name & Contribution Summary | Member Photo | Contribution: Number of Coins |
| :--- | :---: | :---: |
| **Trần Lê Xuân Mai (`24020011`)**<br/>• **Attitude & Self-Leadership:** Took 100% ownership of the project as an individual developer across all 5 weeks (111 total hours invested), attending every weekly tutor review and proactively resolving technical blockers.<br/>• **Technical Knowledge & Implementation:** Built the Roboflow ingestion & MD5 anti-leakage audit (`step1`, `step2`), designed the Controlled HSV/Mosaic training script (`train.py`) reaching `99.5% mAP@50`, and developed the CLI & Live Webcam Web UI (`detect.py`, `test_model.py`, `web_app.py`).<br/>• **Written & Presentation Contribution:** Authored all weekly slide decks (Weeks 1–5), bilingual speaker scripts, the IEEE two-column Final Report, and full GitHub documentation. | *(Insert Member Photo Here)* | **100 / 100 Coins** |

---

# Part 3 — Document Quality Statement

| Quality Aspect | Standard, Benchmark & Verification Evidence |
| :--- | :--- |
| **1. Purpose** | Establish a deterministic, verifiable quality assurance framework across data auditing, `YOLOv11n` model training, evaluation metrics, and IEEE technical documentation so any assessor can reproduce our results. |
| **2. Scope** | Covers all Python modules (`*.py`), dataset configuration (`dataset/data.yaml`), trained weights (`runs/detect/train/weights/best.pt`), audit/evaluation reports (`reports/*.json`, `*.png`), weekly slides, and the IEEE Final Report. |
| **3. Quality Standards** | PEP-8 modular Python code with `pathlib` path safety; IEEE Conference Paper formatting; Model acceptance threshold set at `mAP@50 >= 85%` and `Recall >= 85%` *(Achieved: `99.5% mAP@50`, `98.6% Precision`, `100.0% Recall`)*. |
| **4. Review Process** | 3-gate verification before submission: (1) Clean `.venv` dependency check via `requirements.txt`; (2) Automated data audit (`step2_quality_audit.py`); (3) Independent 24-image test split evaluation (`step5_evaluate_and_inference.py` & `test_model.py`). |
| **5. Data Accuracy** | Purged 100% of empty/invalid `.txt` labels; verified `50.3% Fresh : 49.7% Rotten` class symmetry; confirmed `0` cross-split duplicate images via 128-bit MD5 hashing; locked `hsv_h = 0.005` to preserve true bruise chromaticity. |
| **6. Compliance** | Full compliance with the Faculty of ICT Academic Integrity Policy (all external frameworks properly attributed) and deterministic open-source reproducibility standards (fixed `seed = 0`, saved `args.yaml`). |
| **7. Version Control (Git Insights)** | Tracked in Git under author `Tran Le Xuan Mai <24020011@vnuk.edu.vn>` across 4 structured milestone commits:<br/>• `fa149f6` — `feat(data)`: Roboflow config & MD5 quality audit (`11 files, +1,064 lines`)<br/>• `0aaf526` — `feat(model)`: YOLOv11n AdamW & controlled HSV/Mosaic training (`24 files, +340 lines + best.pt`)<br/>• `e0658ee` — `feat(app)`: CLI inference, evaluation metrics & live Web UI (`35 files, +1,561 lines`)<br/>• `867cb39` — `docs(final)`: IEEE Final Report, weekly slides & personal reflection (`+5,874 lines`) |

