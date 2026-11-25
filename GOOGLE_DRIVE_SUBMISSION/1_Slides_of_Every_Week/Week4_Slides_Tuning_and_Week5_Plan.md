# Bộ Slide Thuyết Trình Tuần 4 & Kế Hoạch Tuần 5
**Môn học:** Đồ án / Chuyên đề Thị giác máy tính & Học sâu
**Đề tài:** Hệ thống Phát hiện và Phân loại Độ tươi Nông sản để Tối ưu Tồn kho (*Automated Food & Produce Freshness Detection System*)

---

## Slide 1: Bức tranh Tổng thể Dự án (Big Picture — Tiến độ đến Tuần 4)

- **Tuần 1 – 2 (Đã hoàn thành):** Xác định bài toán kiểm soát chất lượng nông sản theo chuẩn xuất kho **FIFO**; khảo sát Dataset Roboflow (`>21,000` ảnh); chọn kiến trúc `YOLOv11n` và khu trú phạm vi **6 lớp đại diện** (`Fresh/Rotten` cho Táo, Chuối, Ớt chuông).
- **Tuần 3 (Đã hoàn thành):** Kiểm định chất lượng dữ liệu (*Data Quality Audit*: xóa file `.txt` rỗng, kiểm tra MD5 chống rò rỉ dữ liệu) và huấn luyện mô hình cơ sở **Baseline YOLOv11n** (15 epochs, `mAP@50 = 83.4%`).
- **TUẦN 4 (BÁO CÁO HÔM NAY):**
  1. Tinh chỉnh siêu tham số (**Hyperparameter Tuning**: 50 epochs, `AdamW`, Cosine LR) và áp dụng **Controlled Augmentation** (`hsv_h = 0.005`, `hsv_s = 0.3`, `hsv_v = 0.3`, `mosaic = 1.0`).
  2. Phân tích **Confusion Matrix** để giảm thiểu lỗi bỏ sót quả hỏng (*False Negative*).
  3. Chuyển đổi mã nguồn từ Google Colab về **VS Code Local** (`.venv`) và xây dựng script dự đoán thực tế (`detect.py`, `test_model.py`).
- **TUẦN 5 (KẾ HOẠCH TIẾP THEO):** Tích hợp hoàn chỉnh luồng **End-to-End (`Data -> Model -> Output`)**, đánh giá tổng kết hiệu năng & hạn chế, và rà soát **Final GitHub Review**.

---

## Slide 2: WHAT & WHY — Sản phẩm Bàn giao Tuần 4 & Lý do Khuyến nghị

| Hạng mục Bàn giao Tuần 4 (WHAT) | Lý do Khuyến nghị Kỹ thuật (WHY) |
| :--- | :--- |
| **1. Trọng số Tinh chỉnh (`best.pt` - 5.18 MB)** | Huấn luyện với `AdamW` + Controlled Augmentation giúp nâng `mAP@50` từ `83.4%` lên `93.8%` mà vẫn giữ dung lượng siêu nhẹ cho thiết bị biên. |
| **2. Ma trận Nhầm lẫn (`confusion_matrix.png`)** | Giúp định lượng chính xác tỷ lệ nhầm lẫn giữa các lớp (đặc biệt là giảm lỗi bỏ sót quả hỏng **False Negative**). |
| **3. Dự án VS Code Cục bộ (`train.py`, `.venv`)** | Chuyển từ Colab về VS Code giúp tổ chức mã nguồn chuyên nghiệp, độc lập thư viện hệ thống và dễ dàng phát triển ứng dụng thực tế. |
| **4. Script Kiểm thử Thực tế (`detect.py`, `test_model.py`)** | Cho phép kiểm tra mô hình trực tiếp trên tập ảnh `test/images` chưa từng gặp, tự động vẽ khung Bounding Box và thống kê kết quả. |

---

## Slide 3: WHEN & WHO — Tiến độ Thời gian & Phân công Nhóm (Tuần 4 -> Tuần 5)

| Vai trò trong Nhóm (WHO) | Công việc đã Hoàn thành ở Tuần 4 | Kế hoạch Thực hiện ở Tuần 5 |
| :--- | :--- | :--- |
| **Data Engineer** | Chuẩn hóa `dataset/data.yaml` cho môi trường VS Code cục bộ, bảo toàn phân tách `train/valid/test`. | Đóng gói quy trình kiểm định dữ liệu tự động (`step1`, `step2`) & hoàn thiện tài liệu Dataset. |
| **ML Engineer (Model Lead)** | Tinh chỉnh `YOLOv11n` (50 epochs, `AdamW`, `hsv_h=0.005`, `mosaic=1.0`), xuất trọng số `best.pt`. | Lập bảng tổng kết hiệu năng cuối cùng (**Final Performance Summary**) & phân tích **Limitations / Future Work**. |
| **AI App Developer** | Thiết lập `.venv` trên VS Code, viết `train.py`, `detect.py` và `test_model.py` kiểm thử trên 24 ảnh thực tế. | Tích hợp luồng **Full Pipeline Demo (`Data -> Model -> Output`)** và giao diện trực quan `web_app.py`. |
| **QA & Docs Lead** | Phân tích **Confusion Matrix**, đường cong `PR_curve`, `F1_curve` và đánh giá sai số từng lớp. | Thực hiện **Final GitHub Review** (`Code`, `Documentation`, `Report`, `Reproducibility`) & Slide Tuần 5. |

---

## Slide 4: HOW MUCH — Ngân sách Thời gian, Độ lệch (Gap) & Hành động Khắc phục

| Giai đoạn | Kế hoạch (Planned) | Thực tế (Spent) | Độ lệch (Gap) | Nguyên nhân & Giải pháp của Nhóm (What we are doing about it) |
| :--- | :---: | :---: | :---: | :--- |
| **Tuần 1 – 2: Khảo sát & Thiết kế** | 33h | 34h | **+1h** | Đúng tiến độ; chốt phạm vi 6 lớp nông sản đại diện. |
| **Tuần 3: Data Audit & Baseline** | 22h | 27h | **+5h** | Phát sinh thời gian lọc nhãn `.txt` rỗng và kiểm tra MD5 chống Data Leakage. |
| **Tuần 4: Tuning & VS Code Local** | 25h | 30h | **+5h** | Colab miễn phí giới hạn phiên GPU + thời gian chuyển cấu trúc dự án về VS Code Local. |
| **Lũy kế hết Tuần 4 (W1–W4)** | **80h** | **91h** | **+11h (+13.7%)** | **Hành động khắc phục cho Tuần 5:** Áp dụng quy trình **Hybrid** (gỡ lỗi nhanh 1–2 epochs trên VS Code cục bộ, đẩy `train.py` lên GPU Colab khi train nặng). |
| **Dự kiến Tuần 5 (Final Week)** | **20h** | *Dự kiến 20h* | *0h* | Toàn bộ code nền tảng đã sẵn sàng trên VS Code, đảm bảo về đích đúng hạn. |

---

## Slide 5: Nhìn lại Tuần 3 -> Tuần 4: Top Risks, Major Changes & Key Issues

1. **Top Risks Last Week (Rủi ro lớn nhất tuần qua):**
   - **Biến dạng màu sắc (HSV Color Distortion):** Tăng cường dữ liệu mặc định làm lệch tông màu Hue khiến quả tươi nhìn giống quả thối.
   - **Thiên lệch bối cảnh (Lab Bias):** Ảnh đơn lẻ trong phòng thí nghiệm khó tổng quát hóa khi nhiều quả nằm đè lên nhau trên băng chuyền.
   - **Giới hạn phần cứng cục bộ:** Chạy huấn luyện >21.000 ảnh bằng CPU trên máy cá nhân rất chậm và tốn dung lượng lưu trữ.
2. **Major Changes Last Week (Thay đổi lớn & Phạm vi):**
   - **Chuyển dịch nền tảng phát triển:** Đưa dự án từ Google Colab về **Visual Studio Code Local** với môi trường ảo `.venv` chuẩn hóa.
   - **Khóa chặt tham số Augmentation:** Cố định `hsv_h = 0.005`, `hsv_s = 0.3`, `hsv_v = 0.3` và `mosaic = 1.0` trong `train.py`.
   - **Giữ vững phạm vi (Scope):** Khu trú ở 3 nhóm quả đại diện (Táo, Chuối, Ớt chuông - 6 lớp) để đạt độ chính xác cao nhất cho bản Proof of Concept.
3. **Key Issues Resolved (Vấn đề kỹ thuật đã giải quyết):**
   - Sửa lỗi đường dẫn `train/val/test` trong `dataset/data.yaml` khi chuyển từ Colab về máy tính cá nhân.
   - Cấu hình `.vscode/settings.json` và tự động nhận diện `.venv/bin/python` trong code để không bị lỗi thiếu thư viện khi chạy trên VS Code.

---

## Slide 6: Kết quả Tuần 4 (Phần 1) — Tinh chỉnh Siêu tham số & Controlled Augmentation

- **Cấu hình huấn luyện trong `train.py` & `step4_train_tuned_week4.py`:**
  - Mô hình nền: `YOLOv11n` (`yolo11n.pt`, 2.58M tham số, kích thước ảnh `imgsz = 640`, `batch = 16`).
  - Bộ tối ưu hóa: `AdamW` với tốc độ học khởi tạo `lr0 = 0.001`, giảm dần theo hàm Cosine (`cos_lr = True`) qua `50 epochs`.
  - **Chiến lược Bảo vệ Đặc trưng Màu sắc & Bối cảnh:**
    - `hsv_h = 0.005`: Khóa chặt tông màu để giữ nguyên màu sắc thật của vết thâm đen trên chuối và đốm mốc trên táo/ớt chuông.
    - `hsv_s = 0.3`, `hsv_v = 0.3`: Điều chỉnh nhẹ độ bão hòa và độ sáng mô phỏng ánh đèn kho vận.
    - `mosaic = 1.0`: Ghép 4 ảnh ngẫu nhiên giúp mô hình học cách nhận diện các quả xếp chồng, che khuất nhau.

---

## Slide 7: Kết quả Tuần 4 (Phần 2) — Phân tích Confusion Matrix & Kiểm thử `detect.py`

| Chỉ số Đánh giá (Metric) | Baseline YOLOv11n (Tuần 3 - 15 Epochs) | Tuned YOLOv11n (Tuần 4 - 50 Epochs) | Mức Cải thiện |
| :--- | :---: | :---: | :---: |
| **Precision (Độ chính xác)** | 81.2% | **91.6%** | **+10.4%** |
| **Recall (Độ nhạy - Chống sót hàng hỏng)** | 77.5% | **89.4%** | **+11.9%** |
| **mAP@50** | 83.4% | **93.8%** | **+10.4%** |
| **mAP@50-95** | 61.8% | **74.5%** | **+12.7%** |

- **Kết quả chạy kiểm thử thực tế với `test_model.py` & `detect.py` trên VS Code:**
  - Đã kiểm tra trên **24 ảnh thực tế** trong `test/images` chưa từng gặp khi huấn luyện.
  - Phát hiện chính xác **47 nông sản** (**25 quả Tươi** và **22 quả Hỏng**) đầy đủ cả 6 lớp, xuất ảnh vẽ khung Bounding Box tại `runs/detect/predict/`.

---

## Slide 8: KẾ HOẠCH TUẦN 5 (Week 5 Plan) — Final Integration & Reporting

1. **Mục tiêu 1 — Full Deep Learning Pipeline Demonstration (`Data -> Model -> Output`):**
   - Trình diễn luồng khép kín từ kiểm định dữ liệu (`step2`) $\rightarrow$ huấn luyện mô hình (`train.py`) $\rightarrow$ dự đoán thực tế (`test_model.py`, `detect.py`, `web_app.py`).
2. **Mục tiêu 2 — Final Model Performance Summary:**
   - Tổng hợp báo cáo định lượng cuối cùng (PR Curve, F1 Curve, Confusion Matrix, tốc độ suy luận FPS trên CPU và GPU).
3. **Mục tiêu 3 — Limitations & Risks (Phân tích Hạn chế & Rủi ro):**
   - Đánh giá khách quan giới hạn của góc nhìn đơn 2D (không thấy vết thối ở mặt đáy), hiện tượng thối ngầm bên trong ruột và nhiễu ánh sáng cực đoan.
4. **Mục tiêu 4 — Future Improvements (Đề xuất Hướng Phát triển):**
   - Đề xuất băng chuyền con lăn xoay 360° đa camera, cảm biến cận hồng ngoại (NIR), xuất mô hình sang `ONNX/TensorRT` nhúng lên NVIDIA Jetson Nano và phân hạng FIFO 3 mức (Hạng A/B/C).
5. **Mục tiêu 5 — Final GitHub Review & Reproducibility:**
   - Hoàn thiện kho mã nguồn GitHub đạt 4 tiêu chuẩn: **Code** (module hóa sạch sẽ), **Documentation** (`README.md` chi tiết), **Report** (báo cáo Tuần 1–5) và **Reproducibility** (tái lập 100% kết quả).
   - 👉 **Outcome đích của Tuần 5:** Một giải pháp AI End-to-End hoàn chỉnh kèm tài liệu đầy đủ.
