# Hướng Dẫn Tự Quay Video Demo (2 – 3 Phút) Để Bỏ Vào Thư Mục Này
**Sinh viên:** Trần Lê Xuân Mai (`24020011`)  
**Cách quay nhanh trên máy Mac:** Nhấn tổ hợp phím **`Cmd` + `Shift` + `5`** $\rightarrow$ Chọn **Record Entire Screen (Ghi toàn màn hình)** $\rightarrow$ Nhấn **Record**.

---

## Kịch bản 3 Cảnh Quay (Tổng thời gian ~2 phút):

### Cảnh 1 (0:00 – 0:30): Giới thiệu cấu trúc dự án trên Visual Studio Code
1. Mở cửa sổ **Visual Studio Code** đang hiển thị thư mục `freshness_detection_yolov11`.
2. Nhấp nhanh qua các file chính để thầy thấy cấu trúc module hóa sạch sẽ:
   - `config.py` (cấu hình 6 lớp Táo, Chuối, Ớt chuông & siêu tham số `hsv_h=0.005`, `mosaic=1.0`)
   - `train.py` (huấn luyện Transfer Learning từ `yolo11n.pt` với `AdamW`)
   - `runs/detect/train/results.png` & `confusion_matrix.png` (biểu đồ kết quả huấn luyện 20 epochs)

### Cảnh 2 (0:30 – 1:15): Chạy kiểm tra mô hình trực tiếp trong Terminal của VS Code
1. Mở Terminal trong VS Code (`Cmd` + `~`) và chạy lệnh:
   ```bash
   python3 show_train_results.py
   ```
   *(Để thầy thấy bảng hội tụ 20 epochs và điểm chấm trực tiếp trên tập Test: **Precision 98.6%**, **Recall 100%**, **mAP@50 99.5%**).*
2. Chạy tiếp lệnh dự đoán trên 24 ảnh Test:
   ```bash
   python3 test_model.py
   ```
3. Mở 2–3 bức ảnh kết quả trong `runs/detect/predict/test_produce_0001.jpg`, `test_produce_0004.jpg` để thầy thấy khung Bounding Box màu Xanh lá (Tươi) và màu Đỏ (Hỏng) kèm độ tin cậy `95% – 99%`.

### Cảnh 3 (1:15 – 2:00): Demo Giao diện Web (`web_app.py`) *(Tùy chọn cộng điểm)*
1. Trong Terminal gõ:
   ```bash
   python3 web_app.py
   ```
2. Mở trình duyệt tại `http://localhost:8085`, bấm chọn vài mẫu ảnh `#0001`, `#0004`, `#0012` hoặc tải 1 ảnh lên để hệ thống vẽ khung Bounding Box và bảng thống kê chỉ số màu `H, S, V` ngay lập tức.
3. Kết thúc quay video (`Cmd` + `Ctrl` + `Esc`) và bỏ file `.mp4` / `.mov` vừa quay vào đúng thư mục `2_Video_Demo` này!
