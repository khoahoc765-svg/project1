# Kịch Bản Thuyết Trình Song Ngữ (Việt – Anh) — 11 Trang Slide
**Đề tài:** Automated Food & Produce Freshness Detection System (YOLOv11n)

---

## Trang 1: Bìa báo cáo (Title Slide)
* **Tiếng Việt:** "Em xin chào thầy và các bạn. Hôm nay nhóm em xin trình bày báo cáo Tuần 4 – giai đoạn Tích hợp hoàn chỉnh và Báo cáo tổng kết (Final Integration & Reporting) cho đề tài: Hệ thống Phát hiện và Phân loại Độ tươi Nông sản Tự động. Hệ thống được xây dựng trên kiến trúc lõi YOLOv11n, kết hợp kỹ thuật tăng cường dữ liệu kiểm soát màu sắc HSV và ghép ảnh Mosaic, đồng thời đã được triển khai hoàn chỉnh trên giao diện Web và dòng lệnh (CLI) cục bộ."
* **Tiếng Anh:** "Hello Professor and everyone. Today, our team will present our Week 4 report—Final Integration and Reporting—for the Automated Food and Produce Freshness Detection System. Our solution is built on the YOLOv11n core architecture with controlled HSV and Mosaic augmentation, fully deployed on both a local Web interface and CLI."

---

## Trang 2: Bức tranh tổng thể dự án (Project Big Picture)
* **Tiếng Việt:** "Nhìn vào bức tranh tổng thể từ đầu đến cuối (End-to-End):
  1. **Về vấn đề thực tế:** Quy trình phân loại nông sản thủ công hiện tại đang tạo ra điểm nghẽn lớn, vừa chậm vừa dễ sai sót, gây thất thoát thực phẩm.
  2. **Về giải pháp AI:** Nhóm sử dụng mô hình YOLOv11n để tự động khoanh vùng vị trí và phân loại 6 lớp trạng thái Tươi (Fresh) và Hỏng (Rotten).
  3. **Về kết quả đầu ra:** Hệ thống đưa ra cảnh báo thời gian thực trên băng chuyền để kiểm soát quy trình xuất kho FIFO.
  Trong phạm vi chứng minh khái niệm (PoC), nhóm tập trung vào 3 loại nông sản đại diện nhân với 2 trạng thái (Tươi/Hỏng) tạo thành 6 lớp: Táo, Chuối và Ớt chuông."
* **Tiếng Anh:** "Looking at our end-to-end solution overview:
  1. **Practical Problem:** Manual sorting creates a bottleneck where slow checks lead to food waste.
  2. **AI Core:** We use YOLOv11n for 6-class detection to localize and grade Fresh versus Rotten produce.
  3. **Target Outcome:** Real-time FIFO control with automated conveyor belt alerts.
  Our PoC scope covers 3 representative produce types across 2 states, resulting in 6 classes: Fresh and Rotten Apple, Banana, and Bell Pepper."

---

## Trang 3: Sản phẩm bàn giao & Lý do lựa chọn (Deliverables & Justification)
* **Tiếng Việt:** "Về mục **WHAT (Sản phẩm bàn giao)** và **WHY (Lý do chiến lược)**:
  1. **Bộ dữ liệu đã kiểm định:** Hơn 21.000 ảnh từ Roboflow chia thành 6 lớp. *Lý do:* 3 loại quả này đại diện cho 3 hình mẫu thị giác khác nhau gồm vỏ bóng (Táo), đốm thâm (Chuối) và hình khối phức tạp (Ớt chuông).
  2. **Trọng số mô hình đã tinh chỉnh:** File `best.pt` của YOLOv11n sau 50 vòng lặp (epochs). *Lý do:* Mô hình siêu nhẹ chỉ có 2.58 triệu tham số, đảm bảo tốc độ xử lý thời gian thực trên các thiết bị biên. Khâu tăng cường dữ liệu cũng được khóa thông số sắc độ `hsv_h = 0.005` để giữ nguyên màu sắc thật của vết thối.
  3. **Bộ công cụ thực thi cục bộ:** Gồm `train.py`, `detect.py` và `web_app.py`.
  4. **Kho lưu trữ tái lập:** Toàn bộ mã nguồn GitHub và báo cáo kỹ thuật. *Lý do:* Nhóm áp dụng quy trình lai (Hybrid Workflow) — tận dụng GPU trên Google Colab để huấn luyện nặng và chạy ứng dụng cục bộ trên VS Code."
* **Tiếng Anh:** "Here is WHAT we deliver and WHY we chose this architecture:
  1. **Audited Dataset:** Over 21,000 Roboflow images across 6 classes, chosen because they represent 3 distinct visual archetypes: glossy skin, dark spots, and complex shapes.
  2. **Tuned Weights:** Our 50-epoch YOLOv11n `best.pt` model. With only 2.58 million parameters, it offers ultra-light real-time edge speed, while locking `hsv_h = 0.005` preserves true rot colors.
  3. **Execution Suite:** Local `train.py`, `detect.py`, and `web_app.py` scripts.
  4. **Reproducible Repo:** Complete GitHub code and technical reports, supported by a hybrid workflow combining Colab GPU training with a local VS Code application."

---

## Trang 4: Lịch trình & Phân công nhiệm vụ (Schedule & Team Accountability)
* **Tiếng Việt:** "Về lộ trình **WHEN**, nhóm đi qua 4 tuần: Tuần 1 khảo sát bài toán và dữ liệu; Tuần 2 phân tích EDA và chọn YOLOv11n; Tuần 3 kiểm định dữ liệu và chạy mô hình cơ sở 15 epochs; và hiện tại ở Tuần 4 là tinh chỉnh lên 50 epochs kèm xây dựng ứng dụng cục bộ. (Sang Tuần 5 cuối kỳ, nhóm sẽ hoàn tất kiểm thử tổng thể, đóng gói báo cáo cuối cùng và bàn giao kho GitHub).
  Về phân công **WHO**:
  - **Data Engineer:** Phụ trách nạp dữ liệu từ Roboflow, kiểm tra chống rò rỉ dữ liệu bằng mã băm MD5 và bàn giao `step1_data.py`, `step2_audit.py`.
  - **ML Engineer:** Phụ trách huấn luyện Baseline và tinh chỉnh 50 epochs với bộ tối ưu AdamW, bàn giao `train.py`, `step4_tuned.py` và trọng số `best.pt`.
  - **App & Integration Dev:** Thiết lập môi trường VS Code cục bộ, xây dựng CLI và giao diện Web trực tiếp qua `detect.py` và `web_app.py`.
  - **QA & Docs Lead:** Phân tích ma trận nhầm lẫn (Confusion Matrix) và đóng gói GitHub với `README.md`, `audit.json` cùng bộ Slide."
* **Tiếng Anh:** "Regarding our roadmap: Week 1 covered problem and dataset survey; Week 2 focused on EDA and selecting YOLOv11n; Week 3 completed data auditing and the 15-epoch baseline; and Week 4 delivers 50-epoch tuning and our local app (leading into Week 5 for final system sign-off and repository submission).
  For team accountability:
  - **Data Engineer:** Handled Roboflow ingestion and MD5 anti-leakage audits via `step1_data.py` and `step2_audit.py`.
  - **ML Engineer:** Conducted baseline and 50-epoch AdamW tuning, delivering `train.py`, `step4_tuned.py`, and `best.pt`.
  - **App & Integration Dev:** Built the local VS Code setup, CLI, and live Web UI in `detect.py` and `web_app.py`.
  - **QA & Docs Lead:** Led Confusion Matrix analysis and GitHub packaging, delivering `README.md`, `audit.json`, and slides."

---

## Trang 5: Quản lý nguồn lực thời gian (Resource Management - HOW MUCH)
* **Tiếng Việt:** "Về ngân sách thời gian từ Tuần 1 đến Tuần 4:
  - **Tuần 1:** Dự kiến 15 giờ, thực tế 15 giờ (không chênh lệch).
  - **Tuần 2:** Dự kiến 18 giờ, thực tế 19 giờ (lệch +1 giờ).
  - **Tuần 3:** Dự kiến 22 giờ, thực tế 27 giờ (lệch +5 giờ).
  - **Tuần 4:** Dự kiến 25 giờ, thực tế 30 giờ (lệch +5 giờ).
  Tổng cộng dự kiến là 80 giờ nhưng thực tế tiêu tốn 91 giờ (và dự kiến dành 20 giờ cho Tuần 5). Để xử lý độ chênh lệch 11 giờ này, nhóm đã tự động hóa hoàn toàn việc quét nhãn rỗng trong `step2_audit.py` và áp dụng quy trình lai kết hợp VS Code Local cùng GPU của Colab trong Tuần 4."
* **Tiếng Anh:** "Looking at our planned versus actual hours across the 4 weeks:
  - **Week 1:** 15 hours planned, 15 spent.
  - **Week 2:** 18 hours planned, 19 spent (+1h gap).
  - **Week 3:** 22 hours planned, 27 spent (+5h gap).
  - **Week 4:** 25 hours planned, 30 spent (+5h gap).
  To resolve the overall gap between 80 planned hours and 91 actual hours (with 20 hours allocated for Week 5), we automated empty label scanning in `step2_audit.py` and adopted a hybrid VS Code Local plus Colab GPU workflow in Week 4."

---

## Trang 6: Nhìn lại tuần qua (Weekly Retrospective)
* **Tiếng Việt:** "Nhìn lại các rủi ro, thay đổi và vấn đề kỹ thuật đã giải quyết:
  1. **Rủi ro hàng đầu (Top Risks):** Sự dịch chuyển màu HSV làm biến dạng màu sắc thật của vết thâm; thiên lệch phòng thí nghiệm (Lab Bias) khi ảnh chỉ chụp đơn lẻ; và nguy cơ rò rỉ dữ liệu giữa các tập đã qua Augmentation.
  2. **Thay đổi lớn (Major Changes):** Chuyển đổi mã nguồn sang dạng module hóa trên VS Code; cố định phạm vi ở 6 lớp PoC; và bổ sung bảng điều khiển Web hỗ trợ webcam trực tiếp.
  3. **Vấn đề đã giải quyết (Issues Resolved):** Khắc phục việc CPU cục bộ chạy chậm bằng cách chỉ chạy thử 2 epochs gỡ lỗi trên máy; tự động hóa đường dẫn động cho file YAML; và tự động xóa sạch các file nhãn rỗng bằng script."
* **Tiếng Anh:** "In our weekly retrospective:
  1. **Top Risks:** HSV shifts distorting true bruise colors, Lab Bias from isolated studio shots, and Data Leakage from augmented split overlaps.
  2. **Major Changes:** Transitioned to modular code in VS Code, locked the 6 PoC classes, and added a live webcam Web UI.
  3. **Issues Resolved:** Overcame slow local CPU speeds using 2-epoch local debugging, implemented dynamic YAML path resolution, and auto-purged empty labels via script."

---

## Trang 7: [Special #1] Luồng kiến trúc Học sâu (Full Deep Learning Pipeline)
* **Tiếng Việt:** "Điểm nhấn kỹ thuật đầu tiên là luồng Pipeline hoàn chỉnh từ Dữ liệu đến Mô hình và Đầu ra:
  - **Giai đoạn 1 – Data Pipeline:** Kiểm tra chống rò rỉ bằng mã MD5, bật ghép ảnh `mosaic = 1.0` và khóa sắc độ `hsv_h = 0.005`.
  - **Giai đoạn 2 – YOLOv11n Core:** Sử dụng kiến trúc gồm các khối C3k2, C2PSA và PANet với 100 lớp, 2.58 triệu tham số, được huấn luyện 50 epochs bằng bộ tối ưu AdamW kết hợp lịch trình giảm tốc độ học Cosine (Cosine LR).
  - **Giai đoạn 3 – Deployment:** Xuất ra file trọng số `best.pt` siêu nhẹ chỉ 5.4 MB để chạy dự đoán hàng loạt qua CLI và phân loại qua webcam thời gian thực trên Web App.
  Đặc biệt, đầu ra trực quan không chỉ vẽ hộp giới hạn phân màu (Xanh lá là Tươi, Đỏ là Hỏng) mà còn hiển thị thông số kênh màu HSV và tỷ lệ phần trăm đốm thâm trên bề mặt theo thời gian thực."
* **Tiếng Anh:** "Special #1 demonstrates our full deep learning pipeline from Data to Model to Output:
  - **Stage 1 – Data Pipeline:** Executes an MD5 zero-leakage check, sets `mosaic = 1.0`, and locks `hsv_h = 0.005`.
  - **Stage 2 – YOLOv11n Core:** Features C3k2, C2PSA, and PANet blocks across 100 layers and 2.58 million parameters, trained for 50 epochs via AdamW with cosine learning rate scheduling.
  - **Stage 3 – Deployment:** Exports a compact 5.4 MB `best.pt` file for batch CLI inference and real-time webcam sorting.
  Our visual output draws color-coded bounding boxes—Green for Fresh, Red for Rotten—alongside real-time HSV channel readouts and surface dark-spot percentages."

---

## Trang 8: [Special #2] Đánh giá định lượng (Quantitative Evaluation)
* **Tiếng Việt:** "Điểm nhấn thứ hai là sự bứt phá về hiệu năng giữa mô hình Baseline ở Tuần 3 và mô hình YOLOv11n đã tinh chỉnh ở Tuần 4:
  - **Precision (Độ chính xác):** Tăng từ 81.2% lên **91.6%** (tăng 10.4%).
  - **Recall (Độ phủ):** Tăng mạnh từ 77.5% lên **89.4%** (tăng 11.9%), giúp giảm thiểu tối đa việc bỏ sót nông sản hỏng.
  - **mAP@50:** Tăng từ 83.4% lên **93.8%** (tăng 10.4%).
  - **mAP@50-95:** Tăng từ 61.8% lên **74.5%** (tăng 12.7%).
  Về tốc độ suy luận, mô hình chỉ mất **3.9 ms** trên GPU T4 (tương đương ~250 FPS) và **35.3 ms** trên CPU (~28 FPS), với dung lượng trọng số giữ nguyên ở mức **5.4 MB**."
* **Tiếng Anh:** "Special #2 highlights the quantitative jump from our Week 3 Baseline to our Week 4 Tuned YOLOv11n model:
  - **Precision:** Increased from 81.2% to **91.6%** (+10.4%).
  - **Recall:** Jumped from 77.5% to **89.4%** (+11.9%), significantly reducing missed rotten items.
  - **mAP@50:** Improved from 83.4% to **93.8%** (+10.4%).
  - **mAP@50-95:** Rose from 61.8% to **74.5%** (+12.7%).
  Inference speed reaches **3.9 ms** on a T4 GPU (~250 FPS) and **35.3 ms** on CPU (~28 FPS), with a compact weight size of just **5.4 MB**."

---

## Trang 9: [Special #3 & 4] Đánh giá hạn chế & Lộ trình phát triển (Critical Evaluation)
* **Tiếng Việt:** "Về các hạn chế hiện tại và hướng phát triển kỹ thuật trong tương lai:
  - **Hạn chế hiện tại:** Thứ nhất là điểm mù góc nhìn đơn 2D (không thấy được vết thối nằm ở mặt dưới của quả); thứ hai là nhạy cảm với ánh sáng chói hoặc bóng đổ gắt trong kho làm ảnh hưởng kênh độ sáng Value (V); thứ ba là hiện tượng thối rữa từ bên trong ruột mà camera RGB thông thường không nhìn xuyên qua vỏ được; và thứ tư là phạm vi hiện tại đang tinh chỉnh trên 3 nhóm nông sản đại diện.
  - **Hướng cải tiến tương lai:** Lắp đặt hệ thống con lăn xoay 360 độ kết hợp đa camera trên băng chuyền; tích hợp cảm biến cận hồng ngoại (NIR / Hyperspectral) để soi vết dập dưới vỏ và thối bên trong; xuất mô hình sang chuẩn TensorRT để nhúng lên NVIDIA Jetson Nano kết nối cảnh báo ERP; và nâng cấp lên hệ thống phân loại FIFO 3 cấp độ: Hạng A (Lưu kho) -> Hạng B (Ưu tiên xuất ngay) -> Hạng C (Loại bỏ)."
* **Tiếng Anh:** "For Special #3 and #4, we critically evaluate current limitations and our future engineering roadmap:
  - **Current Limitations:** 2D single-view blindspots hiding bruises on the bottom of the fruit; extreme glare or harsh shadows affecting the Value channel; internal rot invisible on normal RGB skin surfaces; and a class scope currently tuned on 3 representative groups.
  - **Future Improvements:** Implementing 360-degree multi-camera rollers for full surface scanning; integrating NIR or hyperspectral sensors to detect subsurface bruising and internal rot; exporting to Edge TensorRT on NVIDIA Jetson Nano with ERP alerts; and introducing 3-tier FIFO grading: Grade A (Store), Grade B (Export Now), and Grade C (Reject)."

---

## Trang 10: [Special #5] Tiêu chuẩn kỹ thuật & Tái lập GitHub (Engineering Standards)
* **Tiếng Việt:** "Ở điểm nhấn thứ 5 về rà soát GitHub và khả năng tái lập 100% (Reproducibility), kho mã nguồn `freshness_detection_yolov11` được tổ chức theo 3 tiêu chuẩn:
  1. **Chất lượng mã nguồn module hóa:** Tách biệt hoàn toàn file cấu hình trung tâm (`config.py`), tải dữ liệu (`step1_prepare_data.py`), kiểm định chất lượng (`step2_quality_audit.py`), huấn luyện (`train.py`), suy luận (`detect.py`) và giao diện Web (`web_app.py`).
  2. **Tài liệu & Báo cáo đầy đủ:** File `README.md` hướng dẫn cài đặt từng bước với môi trường ảo `.venv`, kèm theo log kiểm định `audit.json`, biểu đồ Confusion Matrix và đường cong PR.
  3. **Khả năng tái lập tất định (Deterministic Reproducibility):** Cố định `seed = 0`, lưu trữ đầy đủ file cấu hình `args.yaml` và tích hợp script `run_pipeline.py` giúp chạy tự động toàn bộ luồng End-to-End chỉ với 1 cú nhấp chuột."
* **Tiếng Anh:** "Special #5 covers our final GitHub review and 100% reproducibility standards inside the `freshness_detection_yolov11` repository:
  1. **Modular Code Quality:** Cleanly decoupled modules for central configuration (`config.py`), data preparation, quality auditing, training, CLI inference, and the live webcam UI.
  2. **Complete Documentation & Report:** A step-by-step `.venv` setup guide in `README.md`, alongside `audit.json` logs, the Confusion Matrix, and PR curves.
  3. **Deterministic Reproducibility:** Guaranteed via a fixed `seed = 0`, saved `args.yaml` parameters, and an automated 1-click end-to-end runner (`run_pipeline.py`)."

---

## Trang 11: Tổng kết & Sẵn sàng Hỏi - Đáp (Thank You & Q&A Defense)
* **Tiếng Việt:** "Trên đây là toàn bộ giải pháp AI End-to-End hoàn chỉnh của nhóm trong 4 tuần qua. Em xin cảm ơn thầy và các bạn đã lắng nghe. Nhóm đã chuẩn bị sẵn sàng để giải đáp các câu hỏi phản biện ạ!"
  - **Tại sao khóa `hsv_h = 0.005`?** Để bảo toàn sắc độ tự nhiên của vết thâm thối, trong khi vẫn mở `hsv_v = 0.3` để mô hình thích nghi với ánh sáng kho hàng.
  - **Làm sao đảm bảo 0% Data Leakage?** Nhóm giữ nguyên tập chia gốc từ Roboflow và dùng thuật toán băm MD5 xác minh không có bất kỳ bức ảnh nào bị trùng lặp giữa các tập Train, Valid và Test.
  - **Quy trình lai Colab + VS Code hoạt động thế nào?** Nhóm đẩy tác vụ huấn luyện nặng 50 epochs lên GPU của Colab, sau đó đưa trọng số về kết hợp với mã nguồn module hóa và giao diện Web trực tiếp trên VS Code cục bộ.
* **Tiếng Anh:** "That concludes our presentation of our polished, end-to-end AI solution. Thank you for listening, and we are now ready for your questions!"
  - **Why lock `hsv_h = 0.005`?** It preserves natural bruise hues while `hsv_v = 0.3` handles warehouse lighting variations.
  - **How did we achieve zero Data Leakage?** We preserved the original dataset splits and verified zero MD5 hash overlap across partitions.
  - **How does the Colab + VS Code hybrid work?** We run heavy 50-epoch GPU training on Colab while maintaining modular code and our live Web UI locally in VS Code.
