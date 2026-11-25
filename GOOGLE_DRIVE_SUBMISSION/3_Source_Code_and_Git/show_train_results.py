"""
Script hiển thị Kết quả Huấn luyện THỰC TẾ của mô hình YOLOv11n (runs/detect/train/weights/best.pt)
Đọc trực tiếp từ file results.csv do Ultralytics sinh ra và chấm điểm thật trên tập Test.
"""

import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__] + sys.argv[1:])

CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))

import pandas as pd
from ultralytics import YOLO

TRAIN_DIR = BASE_DIR / "runs" / "detect" / "train"
WEIGHTS_PATH = TRAIN_DIR / "weights" / "best.pt"
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
RESULTS_CSV = TRAIN_DIR / "results.csv"


def main():
    if not WEIGHTS_PATH.exists() or not RESULTS_CSV.exists():
        print("❌ Chưa tìm thấy kết quả huấn luyện. Hãy chạy 'python3 train.py' trước!")
        return

    df = pd.read_csv(RESULTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    total_epochs = int(df["epoch"].max())

    print("=" * 78)
    print(f"KẾT QUẢ HUẤN LUYỆN THỰC TẾ YOLOv11n ({total_epochs} EPOCHS - TRANSFER LEARNING)")
    print("=" * 78)
    print(f"Trọng số tốt nhất : {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print("-" * 78)
    print(f"{'Epoch':<8} {'box_loss':<10} {'cls_loss':<10} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<10}")
    print("-" * 78)
    for _, row in df.iterrows():
        ep = int(row["epoch"])
        if ep in (1, 2, 5, 10, 15, total_epochs) or ep == total_epochs:
            print(
                f"{ep:>2}/{total_epochs:<4}  "
                f"{row['train/box_loss']:<10.4f} "
                f"{row['train/cls_loss']:<10.4f} "
                f"{row['metrics/precision(B)']*100:>6.2f}%     "
                f"{row['metrics/recall(B)']*100:>6.2f}%     "
                f"{row['metrics/mAP50(B)']*100:>6.2f}%     "
                f"{row['metrics/mAP50-95(B)']*100:>6.2f}%"
            )

    print("\n" + "=" * 78)
    print("ĐÁNH GIÁ TRỰC TIẾP TRỌNG SỐ best.pt TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 78)
    model = YOLO(str(WEIGHTS_PATH))
    metrics = model.val(data=str(DATA_YAML), split="test", imgsz=256, plots=False, verbose=False)

    print(f" - Precision (Độ chính xác) : {metrics.box.mp * 100:.2f}%")
    print(f" - Recall (Độ nhạy)         : {metrics.box.mr * 100:.2f}%")
    print(f" - mAP@50                   : {metrics.box.map50 * 100:.2f}%")
    print(f" - mAP@50-95                : {metrics.box.map * 100:.2f}%")
    print("\nCác biểu đồ thật do Ultralytics xuất ra nằm tại:")
    for fname in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png", "val_batch0_pred.jpg"]:
        p = TRAIN_DIR / fname
        if p.exists():
            print(f"  + {p}")
    print("=" * 78)


if __name__ == "__main__":
    main()
