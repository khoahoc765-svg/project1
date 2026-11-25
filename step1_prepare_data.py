"""
Bước 1: Tải Dataset từ Roboflow qua API (hoặc khởi tạo bộ dữ liệu mô phỏng chuẩn YOLOv11 để kiểm thử Pipeline)
Nguồn dữ liệu: https://universe.roboflow.com/freshness-detection-vwn1a/freshness-detection-rhrze
"""

import argparse
import os
import random
from pathlib import Path

from config import (
    CLASS_NAMES,
    DATASETS_DIR,
    IMG_SIZE,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
)
import cv2
import numpy as np
import yaml


def download_from_roboflow(api_key: str, target_dir: Path) -> Path:
    """Tải bộ dữ liệu Freshness Detection trực tiếp từ Roboflow bằng API Key."""
    from roboflow import Roboflow

    print(f"[Roboflow] Đang kết nối tới workspace '{ROBOFLOW_WORKSPACE}' / project '{ROBOFLOW_PROJECT}'...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = version.download("yolov11", location=str(target_dir))
    data_yaml_path = Path(dataset.location) / "data.yaml"
    print(f"[Roboflow] Tải thành công! File cấu hình: {data_yaml_path}")
    return data_yaml_path


def _draw_produce_item(img: np.ndarray, cls_id: int, cx: int, cy: int, radius: int, rng: random.Random):
    """
    Vẽ nông sản trực quan với đặc trưng màu sắc & bề mặt đặc thù theo 6 lớp:
    0: Fresh apple (Táo tươi - đỏ tươi căng bóng)
    1: Fresh banana (Chuối tươi - vàng tươi cong đều)
    2: Fresh bell pepper (Ớt chuông tươi - xanh/cam khối múi, có cuống xanh)
    3: Rotten apple (Táo hỏng - nâu sẫm, vết thâm đen, biến dạng)
    4: Rotten banana (Chuối hỏng - vàng sẫm nhiều đốm đen/thâm)
    5: Rotten bell pepper (Ớt chuông hỏng - màu xỉn, đốm mốc trắng/xám, nhăn)
    Trả về (x_min, y_min, x_max, y_max) đã giới hạn trong khung ảnh.
    """
    h, w, _ = img.shape

    if cls_id in (0, 3):  # Apple (Táo)
        is_rotten = cls_id == 3
        base_color = (35, 45, 215) if not is_rotten else (30, 55, 115)  # BGR
        cv2.circle(img, (cx, cy), radius, base_color, -1, cv2.LINE_AA)
        # Cuống táo
        cv2.line(img, (cx, cy - radius), (cx + 4, cy - radius - 14), (30, 70, 90), 4)
        if not is_rotten:
            # Ánh sáng phản chiếu vỏ bóng (Fresh apple)
            cv2.circle(img, (cx - radius // 3, cy - radius // 3), max(5, radius // 5), (180, 190, 255), -1, cv2.LINE_AA)
        else:
            # Vết thâm thối & đốm mốc (Rotten apple)
            for _ in range(rng.randint(4, 7)):
                bx = cx + rng.randint(-radius // 2, radius // 2)
                by = cy + rng.randint(-radius // 2, radius // 2)
                br = rng.randint(radius // 6, radius // 3)
                cv2.circle(img, (bx, by), br, (15, 20, 40), -1, cv2.LINE_AA)
                cv2.circle(img, (bx + 2, by - 2), max(2, br // 2), (90, 100, 95), -1, cv2.LINE_AA)
        x1, y1, x2, y2 = cx - radius, cy - radius - 14, cx + radius, cy + radius

    elif cls_id in (1, 4):  # Banana (Chuối)
        is_rotten = cls_id == 4
        base_color = (30, 225, 245) if not is_rotten else (25, 110, 135)  # BGR
        axes = (int(radius * 1.35), int(radius * 0.52))
        angle = rng.randint(-25, 25)
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, base_color, -1, cv2.LINE_AA)
        # Đầu cuống chuối
        cv2.circle(img, (cx - axes[0] + 5, cy), 6, (25, 85, 60), -1)
        if is_rotten:
            # Đốm đen thâm đặc trưng của chuối hỏng
            for _ in range(rng.randint(8, 14)):
                bx = cx + rng.randint(-axes[0] + 12, axes[0] - 12)
                by = cy + rng.randint(-axes[1] + 6, axes[1] - 6)
                cv2.circle(img, (bx, by), rng.randint(4, 10), (15, 20, 25), -1, cv2.LINE_AA)
        pad_x = int(axes[0] * 1.05)
        pad_y = int(axes[1] * 1.25)
        x1, y1, x2, y2 = cx - pad_x, cy - pad_y, cx + pad_x, cy + pad_y

    else:  # Bell pepper (Ớt chuông: cls_id 2 hoặc 5)
        is_rotten = cls_id == 5
        base_color = (45, 175, 40) if not is_rotten else (35, 90, 65)  # BGR
        lobe_r = int(radius * 0.72)
        # Vẽ 3 múi đặc trưng của ớt chuông
        cv2.circle(img, (cx - lobe_r // 2, cy), lobe_r, base_color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + lobe_r // 2, cy), lobe_r, base_color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy + lobe_r // 4), lobe_r, base_color, -1, cv2.LINE_AA)
        # Cuống ớt chuông dày
        cv2.rectangle(img, (cx - 7, cy - lobe_r - 12), (cx + 7, cy - lobe_r + 5), (30, 120, 30), -1)
        if not is_rotten:
            cv2.line(img, (cx - lobe_r // 3, cy - lobe_r // 2), (cx - lobe_r // 3, cy + lobe_r // 2), (120, 230, 120), 3)
        else:
            # Vết mốc trắng xám và vết lõm thối trên ớt chuông hỏng
            for _ in range(rng.randint(5, 8)):
                bx = cx + rng.randint(-lobe_r, lobe_r)
                by = cy + rng.randint(-lobe_r // 2, lobe_r // 2)
                br = rng.randint(6, 14)
                cv2.circle(img, (bx, by), br, (20, 25, 30), -1, cv2.LINE_AA)
                cv2.circle(img, (bx, by), max(2, br // 2), (175, 185, 180), -1, cv2.LINE_AA)
        x1 = cx - lobe_r - lobe_r // 2
        y1 = cy - lobe_r - 12
        x2 = cx + lobe_r + lobe_r // 2
        y2 = cy + lobe_r + lobe_r // 4

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def generate_sample_dataset(target_dir: Path, seed: int = 42) -> Path:
    """
    Khởi tạo bộ dữ liệu mẫu chuẩn cấu trúc Roboflow YOLOv11 (Train / Valid / Test)
    để chạy kiểm định chất lượng dữ liệu (Data Quality Audit) và huấn luyện trên máy local
    khi chưa nhập Roboflow API Key.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    split_counts = {
        "train": 120,
        "valid": 30,
        "test": 24,
    }

    print(f"[Dataset] Đang khởi tạo bộ dữ liệu chuẩn YOLOv11 tại: {target_dir}")
    for split, count in split_counts.items():
        img_dir = target_dir / split / "images"
        lbl_dir = target_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            # Tạo nền băng chuyền / khay siêu thị với độ sáng ngẫu nhiên nhẹ
            bg_val = rng.randint(205, 240)
            img = np.full((IMG_SIZE, IMG_SIZE, 3), (bg_val - 10, bg_val - 5, bg_val), dtype=np.uint8)
            noise = np.random.randint(-8, 8, (IMG_SIZE, IMG_SIZE, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Mỗi ảnh chứa 1 đến 3 quả nông sản (đảm bảo cân bằng giữa 6 class)
            num_objects = rng.randint(1, 3)
            grid_centers = [
                (rng.randint(140, 240), rng.randint(150, 260)),
                (rng.randint(380, 490), rng.randint(160, 280)),
                (rng.randint(220, 420), rng.randint(390, 490)),
            ]
            rng.shuffle(grid_centers)

            yolo_lines = []
            for obj_i in range(num_objects):
                cls_id = (idx + obj_i) % len(CLASS_NAMES)
                cx, cy = grid_centers[obj_i]
                radius = rng.randint(48, 72)
                x1, y1, x2, y2 = _draw_produce_item(img, cls_id, cx, cy, radius, rng)

                bw = (x2 - x1) / IMG_SIZE
                bh = (y2 - y1) / IMG_SIZE
                bcx = (x1 + x2) / 2.0 / IMG_SIZE
                bcy = (y1 + y2) / 2.0 / IMG_SIZE
                if bw > 0.02 and bh > 0.02:
                    yolo_lines.append(f"{cls_id} {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}")

            img_path = img_dir / f"{split}_produce_{idx:04d}.jpg"
            lbl_path = lbl_dir / f"{split}_produce_{idx:04d}.txt"
            cv2.imwrite(str(img_path), img)
            lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    # Cố tình cài cắm 2 file nhãn rỗng (.txt) và 1 file lỗi tọa độ vào tập train
    # để Bước 2 (Data Quality Audit - Mục 3.1 & 3.3) tự động phát hiện và làm sạch!
    train_img_dir = target_dir / "train" / "images"
    train_lbl_dir = target_dir / "train" / "labels"
    dummy_img = np.full((IMG_SIZE, IMG_SIZE, 3), 220, dtype=np.uint8)

    # 1) File nhãn rỗng
    cv2.imwrite(str(train_img_dir / "noisy_empty_label_01.jpg"), dummy_img)
    (train_lbl_dir / "noisy_empty_label_01.txt").write_text("", encoding="utf-8")

    # 2) File tọa độ vượt ngưỡng [0, 1]
    cv2.imwrite(str(train_img_dir / "noisy_corrupt_bbox_02.jpg"), dummy_img)
    (train_lbl_dir / "noisy_corrupt_bbox_02.txt").write_text("0 1.450000 0.500000 -0.200000 0.300000\n", encoding="utf-8")

    data_yaml = {
        "path": str(target_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    data_yaml_path = target_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"[Dataset] Đã tạo xong bộ dữ liệu chuẩn tại: {data_yaml_path}")
    return data_yaml_path


def prepare_dataset(api_key: str = None, force_sample: bool = False) -> Path:
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if api_key and not force_sample:
        rf_dir = DATASETS_DIR / "roboflow_freshness"
        return download_from_roboflow(api_key, rf_dir)
    else:
        sample_dir = DATASETS_DIR / "freshness_yolov11_dataset"
        return generate_sample_dataset(sample_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải hoặc khởi tạo Dataset Freshness Detection YOLOv11")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API Key (hoặc biến môi trường ROBOFLOW_API_KEY)")
    parser.add_argument("--use-sample", action="store_true", help="Sử dụng bộ dữ liệu mô phỏng 6 lớp để kiểm thử ngay")
    args = parser.parse_args()

    yaml_path = prepare_dataset(api_key=args.api_key, force_sample=args.use_sample)
    print(f"Hoàn tất Bước 1. Đường dẫn data.yaml: {yaml_path}")
