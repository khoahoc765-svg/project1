"""
Bước 3: Huấn luyện Mô hình Cơ sở (Baseline Model - Tuần 3, Mục 3.4)
Kiến trúc: YOLOv11n (yolo11n.pt)
"""

import argparse
from pathlib import Path
from config import BASELINE_CONFIG, MODEL_WEIGHTS
import torch
from ultralytics import YOLO


def get_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_baseline_model(data_yaml_path: Path, epochs: int = None, imgsz: int = None, batch: int = None):
    device = get_device()
    cfg = dict(BASELINE_CONFIG)
    if epochs is not None:
        cfg["epochs"] = epochs
    if imgsz is not None:
        cfg["imgsz"] = imgsz
    if batch is not None:
        cfg["batch"] = batch

    print("=" * 70)
    print(f"HUẤN LUYỆN MÔ HÌNH BASELINE YOLOv11n (TUẦN 3) | Device: {device}")
    print(f"Cấu hình: epochs={cfg['epochs']}, imgsz={cfg['imgsz']}, batch={cfg['batch']}")
    print("=" * 70)

    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=str(data_yaml_path),
        device=device,
        workers=2,
        **cfg,
    )
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"[Baseline] Hoàn tất huấn luyện! Trọng số tốt nhất lưu tại: {best_pt}")
    return best_pt, results


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parent / "dataset" / "data.yaml"
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline YOLOv11n (Tuần 3)")
    parser.add_argument("--data", type=str, default=str(default_data), help="Đường dẫn tới file data.yaml")
    parser.add_argument("--epochs", type=int, default=BASELINE_CONFIG["epochs"], help="Số epochs huấn luyện (Mặc định báo cáo: 15)")
    parser.add_argument("--imgsz", type=int, default=BASELINE_CONFIG["imgsz"], help="Kích thước ảnh chuẩn hóa (Mặc định: 640)")
    parser.add_argument("--batch", type=int, default=BASELINE_CONFIG["batch"], help="Batch size")
    args = parser.parse_args()

    train_baseline_model(Path(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
