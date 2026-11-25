"""
Bước 4: Huấn luyện Mô hình Tinh chỉnh & Kiểm soát Augmentation (Tuần 4 - Mục 3.2 & 3.4)
Áp dụng:
- Cố định tông màu (hsv_h = 0.005) để bảo toàn màu sắc thật của vết thối/mốc.
- Điều chỉnh độ sáng (hsv_v = 0.3) mô phỏng ánh đèn kho bảo quản.
- Bật Mosaic = 1.0 mô phỏng các quả xếp chồng che khuất nhau, khắc phục Lab Bias.
- Tinh chỉnh Hyperparameter với AdamW + Cosine Annealing LR.
"""

import argparse
from pathlib import Path
from config import MODEL_WEIGHTS, TUNED_WEEK4_CONFIG
from ultralytics import YOLO

from step3_train_baseline import get_device


def train_tuned_model(data_yaml_path: Path, epochs: int = None, imgsz: int = None, batch: int = None, base_weights: str = None):
    device = get_device()
    cfg = dict(TUNED_WEEK4_CONFIG)
    if epochs is not None:
        cfg["epochs"] = epochs
    if imgsz is not None:
        cfg["imgsz"] = imgsz
    if batch is not None:
        cfg["batch"] = batch

    weights_to_load = base_weights if base_weights else MODEL_WEIGHTS

    print("=" * 70)
    print(f"HUẤN LUYỆN MÔ HÌNH TINH CHỈNH YOLOv11n (TUẦN 4) | Device: {device}")
    print(
        f"Chiến lược Augmentation: hsv_h={cfg['hsv_h']}, hsv_v={cfg['hsv_v']}, "
        f"mosaic={cfg['mosaic']}, degrees={cfg['degrees']}, fliplr={cfg['fliplr']}"
    )
    print(
        f"Siêu tham số (Hyperparameters): optimizer={cfg['optimizer']}, "
        f"lr0={cfg['lr0']}, lrf={cfg['lrf']}, cos_lr={cfg['cos_lr']}, epochs={cfg['epochs']}"
    )
    print("=" * 70)

    model = YOLO(weights_to_load)
    results = model.train(
        data=str(data_yaml_path),
        device=device,
        workers=2,
        **cfg,
    )
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"[Tuned Week 4] Hoàn tất huấn luyện! Trọng số tốt nhất lưu tại: {best_pt}")
    return best_pt, results


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parent / "dataset" / "data.yaml"
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv11n Tinh chỉnh + Controlled Augmentation (Tuần 4)")
    parser.add_argument("--data", type=str, default=str(default_data), help="Đường dẫn tới file data.yaml")
    parser.add_argument("--epochs", type=int, default=TUNED_WEEK4_CONFIG["epochs"], help="Số epochs huấn luyện (Mặc định: 50)")
    parser.add_argument("--imgsz", type=int, default=TUNED_WEEK4_CONFIG["imgsz"], help="Kích thước ảnh chuẩn hóa")
    parser.add_argument("--batch", type=int, default=TUNED_WEEK4_CONFIG["batch"], help="Batch size")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn trọng số khởi tạo (tuỳ chọn)")
    args = parser.parse_args()

    train_tuned_model(
        Path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        base_weights=args.weights,
    )
