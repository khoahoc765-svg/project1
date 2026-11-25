# Kịch Bản Lời Nói Thuyết Trình Chi Tiết (Tuần 4 & Kế Hoạch Tuần 5)
**Đề tài:** Hệ thống Phát hiện và Phân loại Độ tươi Nông sản để Tối ưu Tồn kho (*Automated Food & Produce Freshness Detection System - YOLOv11n*)
**Thời lượng dự kiến:** 7 – 10 phút thuyết trình + 3 – 5 phút Demo & Hỏi đáp (Q&A)

---

## SLIDE 1: Mở đầu & Bức tranh Tổng thể Dự án (Big Picture)
*(Thời lượng: ~1 phút)*

**Lời nói thuyết trình:**
> "Chào thầy và các bạn. Hôm nay nhóm chúng em xin trình bày báo cáo tiến độ **Tuần 4** và kế hoạch về đích **Tuần 5** cho đề tài: **Hệ thống tự động phát hiện và phân loại độ tươi nông sản ứng dụng mô hình YOLOv11n**.
>
> Nhìn vào sơ đồ bức tranh tổng thể (*Big Picture*) trên màn hình:
> - Ở **Tuần 1 và Tuần 2**, nhóm đã xác định bài toán thực tế tại các kho vận và siêu thị: việc kiểm tra rau củ quả thủ công rất chậm và dễ sai sót, gây thất thoát lớn khi áp dụng quy tắc xuất kho **FIFO (Nhập trước – Xuất trước)**. Nhóm đã khảo sát bộ dữ liệu hơn 21.000 ảnh trên Roboflow và chọn kiến trúc siêu nhẹ **YOLOv11n** trên 6 lớp đại diện gồm: **Táo, Chuối và Ớt chuông** ở hai trạng thái Tươi (*Fresh*) và Hỏng (*Rotten*).
> - Ở **Tuần 3**, nhóm đã hoàn tất kiểm định chất lượng dữ liệu (*Data Quality Audit*) và chạy thành công mô hình cơ sở **Baseline 15 epochs**.
> - Bước sang **Tuần 4 hôm nay**, nhóm tập trung vào 3 mục tiêu cốt lõi: **Thứ nhất** là tinh chỉnh siêu tham số và kiểm soát biến đổi màu sắc HSV; **Thứ hai** là phân tích Ma trận nhầm lẫn (*Confusion Matrix*); và **Thứ ba** là chuyển toàn bộ dự án từ Google Colab về chạy cục bộ trên **Visual Studio Code** để kiểm thử thực tế và chuẩn bị cho **Tuần 5 tích hợp toàn diện**."

---

## SLIDE 2: WHAT & WHY — Sản phẩm Bàn giao Tuần 4 & Lý do Khuyến nghị
*(Thời lượng: ~1 phút 15 giây)*

**Lời nói thuyết trình:**
> "Về câu hỏi **WHAT (Tuần này nhóm bàn giao những sản phẩm gì)** và **WHY (Tại sao nhóm lại khuyến nghị các giải pháp đó)**:
>
> Trong Tuần 4, nhóm bàn giao **4 sản phẩm thực tế (Artefacts)**:
> 1. **Một là trọng số mô hình tinh chỉnh `best.pt`** chỉ nặng **5.18 MB**, được huấn luyện với bộ tối ưu `AdamW` và lịch trình giảm tốc độ học Cosine.
> 2. **Hai là bộ biểu đồ đánh giá và Confusion Matrix**, giúp nhóm định lượng chính xác tỷ lệ bỏ sót hàng hỏng (*False Negative*).
> 3. **Ba là môi trường dự án độc lập `.venv` và file huấn luyện chuẩn `train.py`** ngay trên Visual Studio Code.
> 4. **Bốn là bộ script dự đoán và kiểm thử thực tế `detect.py` và `test_model.py`**.
>
> **Tại sao nhóm lại lựa chọn hướng đi này?**
> - Thứ nhất, việc chuyển từ Google Colab về **VS Code Local** giúp mã nguồn được tổ chức chuyên nghiệp, không bị phụ thuộc vào phiên làm việc tạm thời của Colab, sẵn sàng để đóng gói ứng dụng và GitHub vào tuần sau.
> - Thứ hai, nhóm kiên định khu trú ở **3 loại quả đại diện (Táo, Chuối, Ớt chuông)** vì đây là 3 hình mẫu bề mặt hoàn toàn khác nhau: **Táo** có vỏ bóng phản sáng, **Chuối** đặc trưng bởi vết thâm đen đổi màu, còn **Ớt chuông** đại diện cho hình khối nhiều múi và phần cuống xanh. Cách chọn này giúp dữ liệu cân bằng tuyệt đối và chứng minh thuật toán hoạt động hiệu quả nhất (*Proof of Concept*)."

---

## SLIDE 3: WHEN & WHO — Tiến độ Thời gian & Phân công Nhóm
*(Thời lượng: ~1 phút)*

**Lời nói thuyết trình:**
> "Tiếp theo là phần **WHEN (Tiến độ thời gian)** và **WHO (Phân công công việc trong nhóm)**:
>
> Nhóm đang đi đúng lộ trình 5 tuần đã cam kết. Cụ thể trong Tuần 4 và hướng tới Tuần 5, công việc được chia rõ cho 4 vị trí:
> - **Bạn phụ trách Data Engineer:** Đã chuẩn hóa cấu hình `dataset/data.yaml` khi đưa về máy cục bộ, đảm bảo giữ nguyên phân tách `train/valid/test` gốc không bị rò rỉ dữ liệu, và sang Tuần 5 sẽ đóng gói toàn bộ tài liệu kiểm định dữ liệu.
> - **Bạn phụ trách ML Engineer:** Đã trực tiếp thiết lập cấu hình tinh chỉnh 50 epochs với `AdamW` và khóa thông số màu `hsv_h = 0.005`, sang Tuần 5 sẽ phụ trách báo cáo tổng kết hiệu năng và phân tích giới hạn mô hình.
> - **Bạn phụ trách AI App Developer:** Đã thiết lập môi trường ảo `.venv` trên VS Code, viết `train.py`, `detect.py`, `test_model.py` chạy thử nghiệm trên 24 ảnh thực tế, và trong Tuần 5 sẽ hoàn thiện luồng trình diễn End-to-End.
> - **Bạn phụ trách QA & Documentation:** Đã phân tích sâu **Confusion Matrix** ở Tuần 4 và sẽ chịu trách nhiệm rà soát **Final GitHub Review** cùng khả năng tái lập kết quả (*Reproducibility*) ở Tuần 5."

---

## SLIDE 4: HOW MUCH — Ngân sách Thời gian, Độ lệch (Gap) & Giải pháp
*(Thời lượng: ~1 phút)*

**Lời nói thuyết trình:**
> "Trên Slide 4 là biểu đồ **HOW MUCH** — so sánh giữa thời gian kế hoạch và thời gian thực tế nhóm đã bỏ ra:
>
> - Tính lũy kế từ Tuần 1 đến hết Tuần 4, nhóm lên kế hoạch **80 giờ** nhưng thực tế đã dành **91 giờ**, tức là phát sinh độ lệch **+11 giờ** (tương đương +13.7%).
> - **Nguyên nhân của độ lệch +5 giờ ở Tuần 3 và +5 giờ ở Tuần 4** là do: nhóm mất thêm thời gian xử lý các file nhãn `.txt` rỗng, và đặc biệt ở Tuần 4 khi huấn luyện 50 epochs trên hơn 21.000 ảnh, Google Colab bản miễn phí bị giới hạn thời gian GPU còn máy cá nhân chạy bằng CPU lại quá chậm.
> - **Giải pháp nhóm đang thực hiện để xử lý triệt để độ lệch này (What we are doing about it):** Nhóm đã áp dụng **Chiến lược Hybrid (Kết hợp VS Code Local và Colab GPU)**. Cụ thể, trên VS Code cục bộ, nhóm dùng tập dữ liệu nhỏ chạy thử nhanh 1–2 epochs để gỡ lỗi code và xây dựng script kiểm thử `test_model.py` chỉ mất khoảng 1 phút; còn việc train nặng 50 epochs để lấy trọng số `best.pt` điểm cao thì đẩy file `train.py` lên GPU. Nhờ vậy, ở Tuần 5 nhóm hoàn toàn làm chủ tiến độ trong đúng **20 giờ** kế hoạch."

---

## SLIDE 5: Nhìn lại Tuần qua — Top Risks, Major Changes & Key Issues
*(Thời lượng: ~1 phút)*

**Lời nói thuyết trình:**
> "Nhìn lại tuần vừa qua, nhóm tổng kết **3 Rủi ro lớn nhất (Top Risks)**, **3 Thay đổi quan trọng (Major Changes)** và **các Vấn đề kỹ thuật đã giải quyết (Key Issues)**:
>
> 1. **Về Rủi ro (Top Risks):** Nguy hiểm nhất là **biến dạng màu HSV** — nếu để thông số mặc định, thuật toán tăng cường dữ liệu sẽ đổi màu quả táo đỏ tươi sang màu nâu sẫm giống quả thối, làm mô hình học sai. Thứ hai là **thiên lệch phòng thí nghiệm (Lab Bias)** khi ảnh mẫu chỉ chụp từng quả đơn lẻ.
> 2. **Về Thay đổi lớn (Major Changes):** Nhóm quyết định chuyển hẳn mã nguồn từ Notebook rời rạc trên Colab về cấu trúc dự án chuẩn trên **VS Code Local**, đồng thời khóa chặt bộ thông số Augmentation có kiểm soát trong `train.py`.
> 3. **Về Vấn đề kỹ thuật đã xử lý (Key Issues Resolved):** Khi mới đưa dự án về máy Mac, nhóm gặp lỗi đường dẫn tương đối trong `data.yaml` và lỗi VS Code gọi nhầm Python hệ thống thay vì môi trường ảo `.venv`. Nhóm đã khắc phục triệt để bằng cách cấu hình tự động nhận diện `.venv` ngay trong code và chuẩn hóa đường dẫn bằng thư viện `pathlib`."

---

## SLIDE 6: Kết quả Kỹ thuật Tuần 4 (Phần 1) — Tinh chỉnh Siêu tham số & Controlled Augmentation
*(Thời lượng: ~1 phút 15 giây)*

**Lời nói thuyết trình:**
> "Đi sâu vào kết quả kỹ thuật cốt lõi của Tuần 4 trong file `train.py`:
>
> So với bản Baseline Tuần 3 chỉ chạy 15 epochs với thông số mặc định, ở Tuần 4 nhóm thực hiện **2 cải tiến quyết định**:
> - **Thứ nhất là về Tối ưu hóa (Optimization):** Nâng số vòng lặp lên **50 epochs**, sử dụng bộ tối ưu **`AdamW`** với tốc độ học khởi tạo `lr0 = 0.001` kết hợp giảm dần theo hàm **Cosine (`cos_lr = True`)**, giúp hàm Loss hội tụ rất mượt ở giai đoạn cuối.
> - **Thứ hai là Chiến lược Tăng cường dữ liệu có kiểm soát (Controlled Augmentation):**
>   - Nhóm **khóa chặt `hsv_h = 0.005`** để giữ nguyên tông màu gốc của vết thâm đen trên chuối và vết mốc trên táo, ớt chuông.
>   - Nhóm điều chỉnh nhẹ độ bão hòa **`hsv_s = 0.3`** và độ sáng **`hsv_v = 0.3`** để mô hình thích nghi được với ánh đèn vàng hoặc bóng râm trong kho bảo quản.
>   - Đồng thời bật **`mosaic = 1.0`** ghép 4 ảnh lại với nhau, buộc mô hình phải học cách phát hiện chính xác ngay cả khi các quả nằm đè, che khuất một phần của nhau trên băng chuyền."

---

## SLIDE 7: Kết quả Kỹ thuật Tuần 4 (Phần 2) — Confusion Matrix & Kiểm thử `detect.py`
*(Thời lượng: ~1 phút 30 giây — Kết hợp mở VS Code minh họa trực tiếp)*

**Lời nói thuyết trình:**
> "Nhờ chiến lược tinh chỉnh trên, hiệu năng mô hình ở Tuần 4 đã tăng trưởng vượt bậc so với Baseline Tuần 3:
> - Độ chính xác **Precision** tăng từ `81.2%` lên **`91.6%`** (+10.4%), giúp giảm tối đa việc báo hỏng nhầm quả còn tươi.
> - Đặc biệt, chỉ số **Recall (Độ nhạy)** tăng từ `77.5%` lên **`89.4%`** (+11.9%). Trong bài toán kho vận, **Recall là chỉ số sống còn** vì nó chứng minh mô hình đã giảm mạnh lỗi **False Negative** — tức là hạn chế tối đa việc bỏ lọt quả thối vào lô hàng đóng gói.
> - Chỉ số tổng hợp **mAP@50** đạt **`93.8%`** và **mAP@50-95** đạt **`74.5%`**.
>
> Không chỉ dừng ở lý thuyết, nhóm đã viết script `test_model.py` và `detect.py` trên VS Code để kiểm tra mô hình trên **24 bức ảnh thực tế trong tập `test/images`** mà mô hình chưa từng nhìn thấy khi train. Kết quả: mô hình đã phát hiện và khoanh vùng chính xác **47 nông sản** (gồm **25 quả Tươi** được vẽ khung Xanh lá và **22 quả Hỏng** được vẽ khung Đỏ) với tốc độ chỉ khoảng **35 đến 76 mili-giây mỗi ảnh** ngay trên CPU."

---

## SLIDE 8: Kế hoạch Chi tiết Tuần 5 (Week 5 Plan) — Final Integration & Reporting
*(Thời lượng: ~1 phút)*

**Lời nói thuyết trình:**
> "Cuối cùng, trên Slide 8 là **Kế hoạch hành động chi tiết cho Tuần 5 (Final Integration & Reporting)** để đưa dự án về đích:
>
> Trong tuần tới, nhóm tập trung hoàn thiện **5 hạng mục bàn giao cuối cùng**:
> 1. **Một là trình diễn toàn diện luồng học sâu khép kín (`Data -> Model -> Output`):** Kết nối liền mạch từ khâu kiểm định dữ liệu, nạp trọng số `best.pt`, đến xuất kết quả dự đoán trực quan qua CLI và Giao diện Web/Camera thời gian thực.
> 2. **Hai là Báo cáo tổng kết hiệu năng cuối cùng (*Final Model Performance Summary*):** Đóng gói đầy đủ các biểu đồ `PR Curve`, `F1 Curve`, `Confusion Matrix` và bảng benchmark tốc độ FPS.
> 3. **Ba là Phân tích khách quan các Hạn chế & Rủi ro thực tế (*Limitations & Risks*):** Như điểm mù của góc nhìn đơn 2D khi vết thối nằm ở mặt đáy của quả, hoặc hiện tượng quả bị thối ngầm từ bên trong ruột.
> 4. **Bốn là Đề xuất Lộ trình Cải tiến Tương lai (*Future Improvements*):** Như tích hợp băng chuyền con lăn xoay 360 độ đa camera, cảm biến cận hồng ngoại NIR soi dưới lớp vỏ, xuất mô hình sang chuẩn nhẹ `ONNX/TensorRT` nhúng lên mạch NVIDIA Jetson Nano và phân hạng FIFO 3 mức độ (Hạng A lưu kho, Hạng B xuất ngay, Hạng C loại bỏ).
> 5. **Năm là Rà soát Final GitHub Review:** Đảm bảo 4 tiêu chuẩn **Code sạch – Documentation rõ ràng – Report đầy đủ – và Reproducibility tái lập 100% kết quả**.
>
> Bài thuyết trình tiến độ Tuần 4 và kế hoạch Tuần 5 của nhóm xin kết thúc tại đây. Nhóm rất mong nhận được nhận xét và câu hỏi từ thầy ạ!"
