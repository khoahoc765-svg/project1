"""
Script Tổng hợp (End-to-End Pipeline):
Chạy toàn bộ quy trình từ Tuần 1 -> Tuần 4:
1. Tải dữ liệu Roboflow qua API (hoặc khởi tạo bộ dữ liệu chuẩn YOLOv11 6 lớp: Táo, Chuối, Ớt chuông).
2. Kiểm định chất lượng dữ liệu (Data Quality Audit: lọc file .txt rỗng/lỗi tọa độ, kiểm tra MD5 Data Leakage, vẽ biểu đồ phân bố lớp).
3. Huấn luyện Baseline Model YOLOv11n (Tuần 3).
4. Huấn luyện Tuned Model YOLOv11n với Controlled Augmentation (hsv_h=0.005, hsv_v=0.3, mosaic=1.0) & AdamW (Tuần 4).
5. Đánh giá Confusion Matrix, mAP@50, mAP@50-95 và chạy Inference dự đoán ảnh thực tế.
"""

import argparse
import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__] + sys.argv[1:])

from step1_download_or_prepare_data import prepare_dataset
from step2_data_quality_audit import run_data_quality_audit
from step3_train_baseline import train_baseline_model
from step4_train_tuned_week4 import train_tuned_model
from step5_evaluate_and_inference import evaluate_and_run_inference


def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ Pipeline Huấn luyện & Đánh giá YOLOv11n")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API Key (nếu tải bộ 21,000 ảnh từ Roboflow)")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn tới data.yaml có sẵn (nếu đã tải trước đó)")
    parser.add_argument("--baseline-epochs", type=int, default=15, help="Số epochs cho Baseline Model (Tuần 3)")
    parser.add_argument("--tuned-epochs", type=int, default=50, help="Số epochs cho Tuned Model (Tuần 4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh chuẩn hóa")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--skip-baseline", action="store_true", help="Bỏ qua bước train Baseline để chỉ train bản Tinh chỉnh Tuần 4")
    args = parser.parse_args()

    # Bước 1: Chuẩn bị Dataset
    if args.data:
        data_yaml_path = Path(args.data).resolve()
    else:
        data_yaml_path = prepare_dataset(api_key=args.api_key)

    # Bước 2: Kiểm định chất lượng dữ liệu & Làm sạch
    run_data_quality_audit(data_yaml_path, clean_invalid=True)

    # Bước 3: Huấn luyện Baseline Model (Tuần 3)
    best_baseline_pt = None
    if not args.skip_baseline:
        best_baseline_pt, _ = train_baseline_model(
            data_yaml_path=data_yaml_path,
            epochs=args.baseline_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )

    # Bước 4: Huấn luyện Tuned Model + Controlled Augmentation (Tuần 4)
    best_tuned_pt, _ = train_tuned_model(
        data_yaml_path=data_yaml_path,
        epochs=args.tuned_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        base_weights=str(best_baseline_pt) if best_baseline_pt else None,
    )

    # Bước 5: Đánh giá Confusion Matrix & Dự đoán thực tế (Inference)
    evaluate_and_run_inference(
        weights_path=best_tuned_pt,
        data_yaml_path=data_yaml_path,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
