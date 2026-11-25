"""
File kiểm tra nhanh mô hình (test_model.py) ngay trên Terminal của VS Code
(Không cần dùng giao diện Web)
"""

import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__] + sys.argv[1:])

import cv2
from ultralytics import YOLO
from detect import run_inference_on_image

CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))


def main():
    weights_path = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    data_yaml = BASE_DIR / "dataset" / "data.yaml"
    test_images_dir = BASE_DIR / "datasets" / "freshness_yolov11_dataset" / "test" / "images"
    out_dir = BASE_DIR / "runs" / "detect" / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1. KIỂM TRA FILE TRỌNG SỐ MÔ HÌNH (MODEL WEIGHTS)")
    print("=" * 70)
    if not weights_path.exists():
        print(f"❌ Chưa tìm thấy file trọng số tại: {weights_path}")
        print("Hãy chạy 'python train.py' trước!")
        return

    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"✅ Đã tìm thấy trọng số: {weights_path}")
    print(f"✅ Dung lượng mô hình   : {size_mb:.2f} MB")

    model = YOLO(str(weights_path))
    print(f"✅ Các lớp nhận diện (6 classes): {list(model.names.values())}")

    print("\n" + "=" * 70)
    print("2. KIỂM TRA DỰ ĐOÁN TRÊN TẬP ẢNH TEST (24 ẢNH)")
    print("=" * 70)

    test_files = sorted(test_images_dir.glob("*.jpg"))
    class_counts = {}
    fresh_total = 0
    rotten_total = 0

    for img_path in test_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        dets, annotated = run_inference_on_image(model, img_bgr, conf=0.25)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        labels_str = []
        for d in dets:
            cname = d["class_name"]
            class_counts[cname] = class_counts.get(cname, 0) + 1
            if d["is_fresh"]:
                fresh_total += 1
            else:
                rotten_total += 1
            labels_str.append(f"{cname} ({d['confidence']*100:.0f}%)")

        print(f" - {img_path.name}: {', '.join(labels_str) if labels_str else 'Không phát hiện'}")

    print("\n" + "=" * 70)
    print("3. TỔNG KẾT KẾT QUẢ KIỂM TRA MÔ HÌNH")
    print("=" * 70)
    print(f"Tổng số ảnh đã kiểm tra : {len(test_files)} ảnh")
    print(f"Tổng số quả TƯƠI (Fresh) : {fresh_total} quả")
    print(f"Tổng số quả HỎNG (Rotten): {rotten_total} quả")
    print("\nPhân bố chi tiết theo 6 lớp:")
    for cname, cnt in sorted(class_counts.items()):
        print(f"  + {cname:<20}: {cnt} đối tượng")

    print("\n👉 Cách xem ảnh kết quả trực tiếp trong VS Code (không cần Web):")
    print("   Nhìn sang cột Explorer bên trái của VS Code, mở thư mục:")
    print("   runs -> detect -> predict -> Nhấp vào bất kỳ file ảnh .jpg nào để xem khung Bounding Box!")
    print("=" * 70)


if __name__ == "__main__":
    main()
