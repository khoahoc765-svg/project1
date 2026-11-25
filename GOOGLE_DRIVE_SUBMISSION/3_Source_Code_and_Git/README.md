# Hệ thống Tự động Phát hiện và Phân loại Độ tươi Nông sản (YOLOv11n)
**Đề tài:** *Automated Food & Produce Freshness Detection System using YOLOv11*  
**Sinh viên thực hiện:** Trần Lê Xuân Mai (`24020011`)  
**Môn học:** Đồ án / Chuyên đề Trí tuệ Nhân tạo (Thị giác Máy tính & Học sâu)

---

## 1. Tổng quan Dự án (Project Overview)

Trong các chuỗi siêu thị và kho bảo quản nông sản (*Quick-Commerce / Warehousing*), việc phân loại rau củ quả thủ công tốn nhiều nhân lực, tốc độ chậm và dễ bỏ sót hàng hư hỏng, gây khó khăn khi thực thi nguyên tắc xuất kho **FIFO (First-In, First-Out)**.

Dự án xây dựng giải pháp **Object Detection + Freshness Classification** đầu-cuối (*End-to-End*) dựa trên kiến trúc siêu nhẹ **YOLOv11n** (`2.59M` tham số, `6.4 GFLOPs`), thực hiện đồng thời:
1. **Định vị vị trí (Object Localization):** Khoanh vùng chính xác từng quả trên băng chuyền hoặc bàn kiểm định bằng khung giới hạn (`Bounding Box`).
2. **Phân loại trạng thái độ tươi (6 Classes):** Phân biệt trạng thái **Tươi (`Fresh`)** và **Hỏng (`Rotten`)** trên 3 nhóm nông sản đại diện cho 3 hình thái bề mặt thị giác khác nhau:
   - `Fresh Apple` / `Rotten Apple` *(Táo: bề mặt vỏ bóng phản sáng)*
   - `Fresh Banana` / `Rotten Banana` *(Chuối: đặc trưng biến đổi sắc tố vàng sang đốm thâm đen)*
   - `Fresh Bell Pepper` / `Rotten Bell Pepper` *(Ớt chuông: hình khối nhiều múi và cuống xanh)*

---

## 2. Cấu trúc Thư mục Dự án (Repository Structure)

```text
freshness_detection_yolov11/
├── config.py                         # Cấu hình trung tâm (6 lớp, siêu tham số Baseline & Tuned)
├── step1_prepare_data.py             # Tải dữ liệu Roboflow qua API hoặc khởi tạo Dataset chuẩn YOLOv11
├── step2_quality_audit.py            # Data Quality Audit: Lọc .txt rỗng, kiểm tra MD5 chống Data Leakage
├── step3_train_baseline.py           # Huấn luyện mô hình Baseline YOLOv11n
├── step4_train_tuned_week4.py        # Huấn luyện mô hình Tinh chỉnh (AdamW, Cosine LR, Controlled HSV/Mosaic)
├── step5_evaluate_and_inference.py   # Xuất Confusion Matrix, mAP@50, mAP@50-95 & chạy dự đoán tập Test
├── train.py                          # Script huấn luyện chuẩn trên VS Code Local (Transfer Learning từ yolo11n.pt)
├── detect.py                         # Script suy luận CLI vẽ Bounding Box + trích xuất chỉ số màu HSV & tỷ lệ đốm thâm
├── test_model.py                     # Script kiểm tra nhanh trọng số best.pt trên 24 ảnh Test ngay trong Terminal
├── show_train_results.py             # Script in bảng tiến trình hội tụ từ results.csv và chấm điểm trên tập Test
├── web_app.py                        # Giao diện Web & Live Webcam trực quan (http://localhost:8085)
├── run_pipeline.py                   # Script tự động hóa chạy toàn trình từ Bước 1 -> Bước 5 (1-click E2E)
├── requirements.txt                  # Danh sách thư viện chuẩn phục vụ tái lập môi trường (Reproducibility)
├── FINAL_PROJECT_REPORT.md           # Báo cáo kỹ thuật tổng kết toàn dự án (Tuần 1 -> Tuần 5)
├── dataset/
│   └── data.yaml                     # Cấu hình đường dẫn tập Train / Valid / Test và 6 nhãn phân loại
├── reports/
│   ├── audit.json                    # Kết quả kiểm định dữ liệu & kiểm tra chống rò rỉ MD5 (0% leakage)
│   ├── class_distribution.png        # Biểu đồ thống kê phân bố cân bằng nhãn Fresh vs Rotten
│   ├── evaluation_metrics.json       # Báo cáo định lượng Precision, Recall, mAP@50, mAP@50-95 trên tập Test
│   ├── confusion_matrix.png          # Ma trận nhầm lẫn trên tập kiểm thử
│   └── inference_results/            # Ảnh dự đoán thực tế có vẽ khung Bounding Box
└── runs/detect/
    ├── train/                        # Trọng số best.pt (5.21 MB), results.csv, results.png, PR/F1 curves
    └── predict/                      # Kết quả ảnh dự đoán xuất ra từ detect.py / test_model.py
```

---

## 3. Chiến lược Kỹ thuật Cốt lõi (Key Engineering Highlights)

1. **Kiểm định Chất lượng Dữ liệu & Chống Rò rỉ (Data Quality Audit & Zero Data Leakage):**
   - Tự động quét và loại bỏ các file nhãn `.txt` rỗng hoặc sai lệch tọa độ chuẩn hóa `[0, 1]`.
   - Kiểm tra mã băm **MD5** trên toàn bộ ảnh giữa 3 tập `train`, `valid`, `test` (`exact_duplicates_across_splits = 0`), đảm bảo đánh giá khách quan 100%.
2. **Tăng cường Dữ liệu Có Kiểm soát (Controlled HSV & Mosaic Augmentation):**
   - **Khóa sắc độ `hsv_h = 0.005`:** Giữ nguyên tông màu tự nhiên của vết thâm đen và đốm mốc, ngăn hiện tượng đổi màu làm quả tươi biến thành màu quả hỏng.
   - **Điều chỉnh độ sáng `hsv_v = 0.3` & độ bão hòa `hsv_s = 0.3`:** Mô phỏng sự thay đổi ánh đèn trong kho bảo quản và siêu thị.
   - **Bật `mosaic = 1.0` (kết hợp `close_mosaic = 5`):** Ghép 4 ảnh ngẫu nhiên giúp phá vỡ thiên lệch phòng thí nghiệm (*Lab Bias*), huấn luyện mô hình nhận diện tốt khi các quả nằm đè hoặc che khuất nhau trên băng chuyền.
3. **Học chuyển giao & Tối ưu hóa (Transfer Learning & Optimization):**
   - Khởi tạo từ trọng số nền **Pretrained `yolo11n.pt`**, tối ưu bằng **`AdamW`** (`lr0 = 0.001`, `lrf = 0.01`) kết hợp lịch trình giảm tốc độ học hình sin (**Cosine Annealing LR**).

---

## 4. Kết quả Thực nghiệm (Model Performance Summary)

Kết quả đánh giá thực tế của trọng số `runs/detect/train/weights/best.pt` (`5.21 MB`) trên tập **Test độc lập** (`24 ảnh / 47 đối tượng nông sản`):

| Lớp nông sản (Class) | Số lượng mẫu Test | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Trung bình toàn bộ 6 lớp (`all`)** | **47** | **98.6%** | **100.0%** | **99.5%** | **96.7%** |
| `Fresh Apple` (Táo tươi) | 9 | 98.3% | 100.0% | 99.5% | 99.5% |
| `Fresh Banana` (Chuối tươi) | 7 | 100.0% | 100.0% | 99.5% | 92.4% |
| `Fresh Bell Pepper` (Ớt chuông tươi) | 9 | 98.2% | 100.0% | 99.5% | 99.5% |
| `Rotten Apple` (Táo hỏng) | 7 | 98.0% | 100.0% | 99.5% | 99.5% |
| `Rotten Banana` (Chuối hỏng) | 7 | 98.9% | 100.0% | 99.5% | 89.7% |
| `Rotten Bell Pepper` (Ớt chuông hỏng) | 8 | 97.9% | 100.0% | 99.5% | 99.5% |

- **Tốc độ suy luận (Inference Speed):** `~27.4 ms/ảnh` trên CPU (~36 FPS) và `~3.9 ms/ảnh` trên GPU T4 (~250 FPS).
- **Dung lượng trọng số (`best.pt`):** `5.21 MB` (`2,598,266` tham số).

---

## 5. Hướng dẫn Cài đặt & Tái lập Kết quả (Reproducibility Guide)

### Bước 1: Khởi tạo môi trường ảo & Cài đặt thư viện
```bash
cd freshness_detection_yolov11
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Bước 2: Kiểm tra nhanh mô hình đã huấn luyện sẵn (`best.pt`) trên VS Code Terminal
```bash
# 1. Xem bảng tiến trình hội tụ qua các Epochs và điểm mAP trên tập Test:
python3 show_train_results.py

# 2. Chạy dự đoán trên toàn bộ 24 ảnh Test và xuất ảnh vẽ Bounding Box vào runs/detect/predict/:
python3 test_model.py

# 3. Xuất báo cáo đánh giá chi tiết (JSON + Confusion Matrix) vào thư mục reports/:
python3 step5_evaluate_and_inference.py
```

### Bước 3: Chạy dự đoán (Inference) trên ảnh riêng hoặc mở Giao diện Web
```bash
# Dự đoán trên 1 ảnh hoặc thư mục ảnh bất kỳ:
python3 detect.py --source /duong/dan/anh_nong_san.jpg

# Mở Giao diện Web & Live Webcam tại http://localhost:8085:
python3 web_app.py
```

### Bước 4: Huấn luyện lại mô hình từ đầu (Re-training)
```bash
# Huấn luyện với cấu hình chuẩn (Transfer Learning từ yolo11n.pt):
python3 train.py --epochs 20 --imgsz 256 --batch 16

# Hoặc chạy tự động toàn bộ Pipeline từ Bước 1 (Data) -> Bước 5 (Evaluation):
python3 run_pipeline.py --skip-baseline --tuned-epochs 20 --imgsz 256
```
