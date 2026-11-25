# Kịch Bản Chi Tiết Quay Video Demo (2 Phút 30 Giây)
**Sinh viên:** Trần Lê Xuân Mai (`24020011`)  
**Đề tài:** Automated Food & Produce Freshness Detection System using YOLOv11

---

## 🛠️ Bước Chuẩn Bị Trước Khi Bấm Nút Quay (30 giây)
1. Mở sẵn cửa sổ **Visual Studio Code** đang mở thư mục `freshness_detection_yolov11`, bấm `Cmd + ~` để hiện sẵn khung **Terminal** ở dưới.
2. *(Tùy chọn)* Chạy sẵn Web App ở một tab Terminal phụ (`python3 web_app.py`) và mở trình duyệt tại `http://localhost:8085` + mở thêm 1 tab file báo cáo `IEEE_FINAL_REPORT_PAPER.html`.
3. Nhấn tổ hợp phím **`Cmd` + `Shift` + `5`** trên bàn phím Mac:
   - Chọn biểu tượng **Record Entire Screen** (Ghi toàn màn hình).
   - Bấm vào chữ **Options** $\rightarrow$ Phần **Microphone** nhớ chọn **MacBook Microphone** (để thu cả giọng nói thuyết minh của bạn).
   - Bấm **Record** và làm theo đúng 4 phân cảnh dưới đây!

---

## 🎬 KỊCH BẢN QUAY & LỜI NÓI TỪNG PHÚT (2 – 3 PHÚT)

### Phân cảnh 1 (0:00 – 0:35): Giới thiệu Đề tài & Công nghệ sử dụng ("Dùng công cụ gì để làm?")
* **Thao tác trên màn hình:** Chiếu cửa sổ **VS Code** đang mở file `README.md` (hoặc `config.py`), rê chuột dọc theo cây thư mục bên trái.
* **Lời nói (Tiếng Việt):**
  > *"Em chào thầy ạ. Em là Trần Lê Xuân Mai, mã sinh viên 24020011. Hôm nay em xin demo dự án cuối kỳ: Hệ thống Tự động Phát hiện và Phân loại Độ tươi Nông sản. Về công nghệ sử dụng, toàn bộ hệ thống được xây dựng bằng ngôn ngữ **Python** trên **Visual Studio Code** với môi trường ảo `.venv` độc lập, sử dụng framework học sâu **PyTorch** và **Ultralytics YOLOv11**, thư viện xử lý ảnh **OpenCV**, dữ liệu từ **Roboflow**, quản lý phiên bản bằng **Git**, và báo cáo kỹ thuật được soạn theo chuẩn bài báo khoa học 2 cột của **IEEE**."*
* *(Nếu thầy yêu cầu nói Tiếng Anh)*:
  > *"Hello Professor. My name is Tran Le Xuan Mai, student ID 24020011. This is the demo for my final project: Automated Food and Produce Freshness Detection System. Our tech stack is built in Python inside Visual Studio Code, powered by PyTorch, Ultralytics YOLOv11, OpenCV, and Roboflow, tracked with Git, and documented using the two-column IEEE conference paper template."*

---

### Phân cảnh 2 (0:35 – 1:20): Giải thích Thuật toán & Điểm nhấn Kỹ thuật ("Xài thuật toán gì?")
* **Thao tác trên màn hình:** Nhấp chuột mở file `step2_quality_audit.py` (3 giây), sau đó nhấp mở file **`train.py`** và bôi sáng đoạn cấu hình `model.train(...)` từ dòng 43 đến dòng 65.
* **Lời nói (Tiếng Việt):**
  > *"Về mặt thuật toán, hệ thống giải quyết đồng thời bài toán định vị Bounding Box và phân loại 6 lớp trạng thái Tươi / Hỏng cho 3 nhóm nông sản đại diện là Táo, Chuối và Ớt chuông:*
  > *1. Thứ nhất, ở khâu dữ liệu (`step2_quality_audit.py`), em viết thuật toán tự động lọc xóa các file nhãn `.txt` rỗng và dùng hàm băm **MD5 Hash** đối chiếu giữa tập Train, Valid và Test để đảm bảo **0% rò rỉ dữ liệu (Zero Data Leakage)**.*
  > *2. Thứ hai, ở khâu mô hình (`train.py`), em sử dụng kiến trúc mạng nơ-ron siêu nhẹ **YOLOv11n** (chỉ 2.59 triệu tham số) kết hợp **Học chuyển giao (Transfer Learning)** từ trọng số nền `yolo11n.pt` và tối ưu bằng **AdamW** với lịch trình giảm tốc độ học **Cosine Annealing**.*
  > *3. Điểm đặc biệt nhất của thuật toán là chiến lược **Controlled Augmentation**: em khóa chặt sắc độ **`hsv_h = 0.005`** để giữ nguyên màu sắc tự nhiên của các vết thâm đen và đốm mốc, đồng thời bật ghép 4 ảnh **`mosaic = 1.0`** để chống thiên lệch phòng thí nghiệm (Lab Bias), giúp nhận diện tốt các quả nằm đè lên nhau."*
* *(Bản Tiếng Anh ngắn gọn)*:
  > *"For our methodology: First, `step2_quality_audit.py` purges corrupt label files and runs an MD5 hash check to guarantee zero data leakage across splits. Second, in `train.py`, we fine-tune the lightweight YOLOv11n architecture via Transfer Learning from `yolo11n.pt` using the AdamW optimizer and Cosine Annealing LR. Crucially, we lock Hue augmentation at `hsv_h = 0.005` to preserve natural bruise and mold colors, while enabling `mosaic = 1.0` to handle overlapping produce."*

---

### Phân cảnh 3 (1:20 – 2:10): Chạy thực tế Mô hình trên Terminal & Xem Kết quả
* **Thao tác trên màn hình:**
  1. Bấm xuống ô **Terminal** của VS Code, gõ lệnh thứ nhất và nhấn Enter:
     ```bash
     python3 show_train_results.py
     ```
  2. Đợi 3 giây cho bảng điểm hiện ra, sau đó gõ tiếp lệnh thứ hai:
     ```bash
     python3 test_model.py
     ```
  3. Nhìn sang cột bên trái, nhấp mở nhanh `runs/detect/train/results.png`, `confusion_matrix.png`, rồi mở `runs/detect/predict/test_produce_0001.jpg` và `test_produce_0004.jpg`.
* **Lời nói (Tiếng Việt):**
  > *"Bây giờ em sẽ chạy kiểm chứng trực tiếp mô hình `best.pt` trên Terminal. Khi chạy `show_train_results.py` và `test_model.py` trên tập 24 ảnh Test độc lập chưa từng gặp khi train, mô hình phát hiện chính xác tuyệt đối **47 trên 47 quả**, đạt **Precision 98.6%**, **Recall 100%** — tức là không bỏ lọt bất kỳ quả thối hỏng nào, và **mAP@50 đạt 99.5%** với tốc độ xử lý rất nhanh khoảng **27 mili-giây mỗi ảnh** ngay trên CPU.*
  > *Trên màn hình là đồ thị hội tụ `results.png`, ma trận nhầm lẫn `confusion_matrix.png` và các ảnh dự đoán thực tế: quả tươi được khoanh khung màu Xanh lá, quả hỏng được khoanh khung màu Đỏ với độ tin cậy từ 90% đến 99%."*
* *(Bản Tiếng Anh ngắn gọn)*:
  > *"Now let's run `show_train_results.py` and `test_model.py` in the terminal on the unseen 24-image test split. Our 5.21 MB `best.pt` model accurately detects all 47 out of 47 items, achieving 98.6% Precision, 100% Recall with zero missed rotten produce, and 99.5% mAP@50 at 27.4 milliseconds per image on CPU. Here are our convergence curves, confusion matrix, and predicted bounding boxes—green for Fresh and red for Rotten."*

---

### Phân cảnh 4 (2:10 – 2:45): Trình diễn Giao diện Web (`web_app.py`) & Chốt bài
* **Thao tác trên màn hình:** Chuyển qua trình duyệt đang mở `http://localhost:8085`, nhấp chuột vào 3–4 nút **Mẫu #0001, Mẫu #0003, Mẫu #0012** ở cột bên trái (hoặc bấm "Tải ảnh nông sản từ máy tính"), sau đó chuyển nhanh sang tab bài báo 2 cột `IEEE_FINAL_REPORT_PAPER.html`.
* **Lời nói (Tiếng Việt):**
  > *"Cuối cùng, em đã tích hợp trọng số `best.pt` vào ứng dụng Web cục bộ `web_app.py` hỗ trợ cả tải ảnh lên và quét qua Webcam trực tiếp. Ngoài việc vẽ Bounding Box phân loại Tươi và Hỏng, hệ thống còn trích xuất thêm thông số kênh màu **H, S, V** và tính **tỷ lệ phần trăm đốm thâm** trên bề mặt quả để phục vụ bài toán xuất kho FIFO. Bên cạnh đó, toàn bộ kết quả thực nghiệm đã được trình bày đầy đủ trong báo cáo cuối kỳ theo đúng mẫu 2 cột của IEEE. Em xin cảm ơn thầy đã theo dõi phần demo của em ạ!"*
* *(Bản Tiếng Anh ngắn gọn)*:
  > *"Finally, here is our local Web and Live Webcam dashboard (`web_app.py`), which allows uploading images or streaming live camera frames, displaying bounding boxes alongside HSV color statistics and dark-spot percentages for FIFO inventory control, accompanied by our IEEE two-column final paper. Thank you for watching my demo!"*

---

## 💾 Kết Thúc & Lưu Video
- Nhấn **`Cmd` + `Ctrl` + `Esc`** (hoặc bấm nút Stop trên thanh menu góc phải trên cùng màn hình Mac) để dừng quay.
- Đổi tên video thành `Demo_Freshness_Detection_YOLOv11_TranLeXuanMai_24020011.mov` và kéo bỏ vào thư mục `GOOGLE_DRIVE_SUBMISSION/2_Video_Demo/` trước khi tải lên Google Drive!
