# BÁO CÁO KỸ THUẬT TỔNG KẾT ĐỒ ÁN TRÍ TUỆ NHÂN TẠO (TUẦN 1 – TUẦN 5)
**Đề tài:** Hệ thống Tự động Phát hiện và Phân loại Độ tươi Nông sản để Tối ưu Tồn kho (*Automated Food & Produce Freshness Detection System using YOLOv11*)  
**Sinh viên thực hiện:** Trần Lê Xuân Mai (`24020011`)  
**Nhóm:** Group 3 – Artificial Intelligence 2026

---

## 1. Đặt vấn đề & Mục tiêu Hệ thống (Problem Statement & Objectives)

### 1.1 Bài toán thực tế
Tại các hệ thống siêu thị, trung tâm phân phối và kho vận nông sản (*Quick-Commerce & Cold-Chain Warehousing*), quy trình phân loại chất lượng rau củ quả hiện nay chủ yếu dựa vào nhân công kiểm tra bằng mắt thường. Phương pháp thủ công này tồn tại 3 hạn chế lớn:
1. **Tốc độ xử lý chậm và thiếu nhất quán** khi lưu lượng hàng hóa lớn trên băng chuyền.
2. **Tỷ lệ bỏ sót nông sản hư hỏng (False Negative) cao**, khiến nấm mốc lây lan sang các lô hàng lành lặn.
3. **Khó thực thi chính xác quy tắc xuất kho FIFO (First-In, First-Out)** theo mức độ tươi thực tế của nông sản.

### 1.2 Mục tiêu giải pháp AI
Xây dựng hệ thống Thị giác máy tính kết hợp Học sâu (**YOLOv11n**) hoạt động theo thời gian thực để:
- **Định vị đối tượng (Object Detection):** Xác định tọa độ khung giới hạn (`Bounding Box`) của từng quả trong khung hình.
- **Phân loại trạng thái (Freshness Classification):** Phân biệt chính xác 6 lớp trạng thái **Tươi (`Fresh`)** và **Hỏng (`Rotten`)** trên 3 nhóm nông sản đại diện:
  - `Fresh Apple` (Táo tươi) & `Rotten Apple` (Táo hỏng)
  - `Fresh Banana` (Chuối tươi) & `Rotten Banana` (Chuối hỏng)
  - `Fresh Bell Pepper` (Ớt chuông tươi) & `Rotten Bell Pepper` (Ớt chuông hỏng)

### 1.3 Cơ sở chiến lược khi chọn 3 nhóm nông sản đại diện (Proof of Concept Rationale)
Việc tập trung vào Táo, Chuối và Ớt chuông mang ý nghĩa khoa học và kỹ thuật rõ ràng:
- **Đại diện cho 3 hình mẫu bề mặt thị giác (3 Visual Archetypes):** Táo có vỏ cầu nhẵn bóng dễ phản xạ ánh sáng; Chuối có dáng cong dài và đặc trưng hư hỏng qua sự biến đổi sắc tố vàng sang đốm đen; Ớt chuông có cấu trúc hình khối nhiều múi kèm phần cuống xanh phức tạp.
- **Đảm bảo cân bằng dữ liệu (Class Balance):** Tránh hiện tượng suy giảm `mAP` do mất cân bằng mẫu khi gộp quá nhiều loại quả có số lượng ảnh ít.

---

## 2. Kiểm định Chất lượng Dữ liệu & Tiền xử lý (Data Quality Audit & Preprocessing)

Bộ dữ liệu được chuẩn hóa theo định dạng YOLOv11 (`dataset/data.yaml`) và đi qua quy trình kiểm định tự động trong `step2_quality_audit.py`:

| Tiêu chí Kiểm định | Rủi ro Kỹ thuật | Giải pháp Đã Triển khai trong Dự án | Kết quả Kiểm chứng |
| :--- | :--- | :--- | :--- |
| **1. Tính hợp lệ (Validity)** | File nhãn `.txt` rỗng hoặc tọa độ Box nằm ngoài `[0, 1]` gây lỗi hàm Loss. | Script `step2_quality_audit.py` tự động quét và xóa sạch các cặp ảnh – nhãn lỗi trước khi đưa vào huấn luyện. | Đã lọc sạch 100% file nhãn lỗi/rỗng; giữ lại 174 ảnh hợp lệ (`120 train`, `30 valid`, `24 test`). |
| **2. Tính cân bằng (Class Balance)** | Lệch tỷ lệ giữa nhóm `Fresh` và `Rotten` gây thiên lệch dự đoán. | Thống kê số lượng Bounding Box trên toàn bộ 6 lớp và xuất biểu đồ `reports/class_distribution.png`. | Tỷ lệ cân bằng đồng đều: **50.3% Fresh : 49.7% Rotten** (367 tổng số Bounding Box). |
| **3. Chống Rò rỉ Dữ liệu (Anti-Data Leakage)** | Ảnh trùng lặp xuất hiện ở cả tập `train` và `test` làm điểm `mAP` ảo. | Thuật toán băm **MD5 Hash** đối chiếu từng byte của mọi bức ảnh giữa 3 tập `train / valid / test`. | **0 ảnh trùng lặp** (`exact_duplicates_across_splits = 0`, đạt chuẩn `PASSED`). |
| **4. Bảo toàn Đặc trưng Màu sắc (HSV Control)** | Biến đổi kênh Hue (`hsv_h`) quá mức làm quả tươi đổi sang màu thâm của quả thối. | Khóa chặt **`hsv_h = 0.005`**, chỉ cho phép biến thiên nhẹ độ bão hòa **`hsv_s = 0.3`** và độ sáng **`hsv_v = 0.3`**. | Giữ nguyên màu sắc thật của vết thâm/mốc, đồng thời thích nghi tốt với ánh đèn kho. |
| **5. Khắc phục Lab Bias (Mosaic Augmentation)** | Ảnh chụp đơn lẻ không phản ánh tình huống các quả nằm đè lên nhau trên băng chuyền. | Bật **`mosaic = 1.0`** ghép 4 ảnh ngẫu nhiên trong 15 epochs đầu và tắt ở 5 epochs cuối (`close_mosaic = 5`). | Mô hình định vị chính xác ngay cả khi nhiều quả xếp gần hoặc che khuất một phần. |

---

## 3. Kiến trúc Mô hình & Cấu hình Huấn luyện (Model Architecture & Training Setup)

- **Kiến trúc mạng:** `YOLOv11n` (Nano) gồm **101 layers** sau khi hợp nhất (*fused*), **2,598,266 tham số**, độ phức tạp tính toán chỉ **6.4 GFLOPs**.
  - **Backbone (`C3k2` + `SPPF` + `C2PSA`):** Trích xuất đặc trưng đa tỉ lệ kết hợp cơ chế chú ý không gian (*Spatial Attention*) giúp nhận diện các đốm thối/mốc nhỏ trên vỏ quả.
  - **Neck (`PANet`):** Tổng hợp đặc trưng hình học và màu sắc giữa các tầng sâu và nông.
  - **Decoupled Detection Head:** Tách riêng nhánh hồi quy tọa độ hộp (`box_loss` + `dfl_loss`) và nhánh phân loại 6 lớp (`cls_loss`).
- **Siêu tham số huấn luyện (`train.py`):**
  - Trọng số khởi tạo: **Pretrained `yolo11n.pt`** (Transfer Learning).
  - Bộ tối ưu hóa: **`AdamW`** (`lr0 = 0.001`, `lrf = 0.01`, `weight_decay = 0.0005`, `cos_lr = True`, `warmup_epochs = 2.0`).
  - Cố định hạt giống ngẫu nhiên: `seed = 0` để đảm bảo tính tái lập 100% (*Deterministic Reproducibility*).

---

## 4. Kết quả Huấn luyện & Đánh giá Định lượng (Quantitative Evaluation)

### 4.1 Tiến trình Hội tụ qua 20 Epochs Huấn luyện Thực tế (`runs/detect/train/results.csv`)

| Epoch | Train Box Loss | Train Cls Loss | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 / 20** | 1.3802 | 3.3106 | 62.66% | 34.48% | 62.62% | 48.03% |
| **5 / 20** | 0.6746 | 1.3095 | 90.02% | 94.37% | 96.79% | 81.91% |
| **10 / 20** | 0.5621 | 0.8712 | 97.69% | 98.48% | 99.50% | 90.09% |
| **15 / 20** | 0.4910 | 0.6694 | 98.60% | 100.00% | 99.50% | 93.36% |
| **20 / 20** | **0.3256** | **0.3820** | **98.90%** | **100.00%** | **99.50%** | **96.51%** |

### 4.2 Kết quả Kiểm định Độc lập trên Tập Test (`24 ảnh / 47 đối tượng`)

| Lớp nông sản (Class) | Số mẫu Test | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Trung bình toàn cục (`all`)** | **47** | **98.6%** | **100.0%** | **99.5%** | **96.7%** |
| `Fresh Apple` | 9 | 98.3% | 100.0% | 99.5% | 99.5% |
| `Fresh Banana` | 7 | 100.0% | 100.0% | 99.5% | 92.4% |
| `Fresh Bell Pepper` | 9 | 98.2% | 100.0% | 99.5% | 99.5% |
| `Rotten Apple` | 7 | 98.0% | 100.0% | 99.5% | 99.5% |
| `Rotten Banana` | 7 | 98.9% | 100.0% | 99.5% | 89.7% |
| `Rotten Bell Pepper` | 8 | 97.9% | 100.0% | 99.5% | 99.5% |

- **Chỉ số Recall đạt 100% trên nhóm `Rotten`:** Đảm bảo không bỏ lọt bất kỳ sản phẩm hư hỏng nào trong tập kiểm thử (*Zero False Negatives*).
- **Tốc độ xử lý:** Trung bình **27.4 ms/ảnh trên CPU** (~36 FPS), hoàn toàn đáp ứng yêu cầu quét băng chuyền thời gian thực mà không bắt buộc phải có card đồ họa rời đắt tiền tại điểm triển khai.

---

## 5. Triển khai Ứng dụng & Phân tích Mở rộng (Deployment & Visual Analytics)

Hệ thống được tích hợp sẵn 2 phương thức vận hành thực tế:
1. **Công cụ Dòng lệnh (`detect.py` & `test_model.py`):** Quét hàng loạt hình ảnh lô hàng, vẽ khung giới hạn phân màu trực quan (**Xanh lá = Tươi**, **Đỏ = Hỏng**) và lưu tại `runs/detect/predict/`.
2. **Giao diện Web & Live Webcam (`web_app.py`):** Khởi chạy máy chủ cục bộ tại `http://localhost:8085`, hỗ trợ tải ảnh, chọn nhanh 24 mẫu kiểm thử hoặc bật Webcam trực tiếp. Hệ thống đồng thời trích xuất thông số kênh màu **`H, S, V`** và **tỷ lệ phần trăm đốm thâm (`dark_spot_ratio %`)** bên trong từng khung Bounding Box.

---

## 6. Hạn chế Hiện tại & Hướng Phát triển Tương lai (Limitations & Future Work)

### 6.1 Hạn chế & Rủi ro kỹ thuật
1. **Điểm mù góc nhìn đơn 2D (*Single-View Blindspot*):** Một camera đơn lắp cố định phía trên không thể quan sát được vết thối/dập nếu nó nằm hoàn toàn ở mặt đáy của quả tiếp xúc với băng chuyền.
2. **Hư hỏng ngầm bên trong (*Internal Rot*):** Trường hợp quả bị thối từ phần lõi nhưng lớp vỏ ngoài chưa đổi màu trên phổ ánh sáng khả kiến RGB.
3. **Nhiễu ánh sáng cực đoan:** Hiện tượng lóa sáng (*Specular Glare*) quá gắt trên vỏ quả bóng có thể ảnh hưởng cục bộ đến kênh độ sáng `Value (V)`.

### 6.2 Lộ trình Cải tiến
1. **Cơ cấu Băng chuyền Con lăn Xoay 360° Đa Camera:** Giúp quét toàn bộ bề mặt nông sản từ nhiều góc độ.
2. **Kết hợp Cảm biến Cận hồng ngoại (*NIR / Hyperspectral Imaging*):** Phát hiện độ ẩm và sự phân hủy dưới lớp vỏ trước khi vết thối lộ ra ngoài.
3. **Triển khai Edge AI (`ONNX` / `TensorRT`):** Nhúng trọng số `best.pt` (`5.21 MB`) lên thiết bị biên NVIDIA Jetson Nano đặt trực tiếp tại kho.
4. **Phân hạng FIFO 3 Cấp độ kết nối ERP:** Kết hợp nhãn YOLOv11n và tỷ lệ đốm thâm (`dark_spot_ratio %`) để tự động phân luồng: **Hạng A** (*Tươi nguyên – Lưu kho dài hạn*) $\rightarrow$ **Hạng B** (*Chín tới – Ưu tiên xuất kho FIFO ngay*) $\rightarrow$ **Hạng C** (*Hư hỏng – Loại bỏ*).
