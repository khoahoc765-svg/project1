import argparse
import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__] + sys.argv[1:])

import torch
from ultralytics import YOLO

CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))


def get_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv11 nhận diện độ tươi nông sản")
    parser.add_argument("--epochs", type=int, default=15, help="Số vòng lặp huấn luyện (mặc định: 15)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh chuẩn hóa (mặc định: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (mặc định: 16)")
    args = parser.parse_args()

    # 1. Khởi tạo mô hình nền siêu nhẹ (Nano)
    weights_path = BASE_DIR / "yolo11n.pt"
    model = YOLO(str(weights_path) if weights_path.exists() else "yolo11n.yaml")

    device = get_device()
    print(f"Đang chạy huấn luyện trên thiết bị: {device}")

    # 2. Cấu hình các siêu tham số huấn luyện
    results = model.train(
        data=str(BASE_DIR / "dataset" / "data.yaml"),  # Đường dẫn tới file cấu hình dữ liệu
        epochs=args.epochs,         # Số vòng lặp huấn luyện (tăng lên 50-100 khi train thật trên GPU)
        imgsz=args.imgsz,           # Kích thước ảnh chuẩn hóa
        batch=args.batch,           # Số lượng ảnh đẩy vào mỗi cụm

        # Bộ tối ưu hóa (AdamW + Cosine Annealing LR)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2.0,
        seed=0,

        # Các thông số Augmentation bảo vệ đặc trưng màu sắc (như báo cáo Tuần 3 & 4)
        mosaic=1.0,                 # Bật ghép ảnh chống Lab Bias
        close_mosaic=5,             # Tắt mosaic ở 5 epoch cuối để bám sát viền Bounding Box
        hsv_h=0.005,                # Khóa chặt tông màu bảo tồn vết thâm/mốc
        hsv_s=0.3,                  # Điều chỉnh nhẹ độ bão hòa
        hsv_v=0.3,                  # Điều chỉnh nhẹ độ sáng (mô phỏng ánh đèn)

        # Thiết bị chạy (tự động chọn 'mps' trên Mac Apple Silicon hoặc 'cpu')
        device=device,
        project=str(BASE_DIR / "runs" / "detect"),
        name="train",
        exist_ok=True,
    )

    print(f"Huấn luyện hoàn tất! Trọng số tốt nhất được lưu tại: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
