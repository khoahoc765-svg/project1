"""
Bước 2: Kiểm định Chất lượng Dữ liệu & Tiền xử lý Làm sạch (Data Quality Audit & Cleaning)
Thực hiện đúng theo Báo cáo Tuần 2 (Mục 2.3) & Tuần 3 (Mục 3.1, 3.3):
1. Tính hợp lệ (Validity): Quét tự động và loại bỏ file nhãn .txt rỗng hoặc lỗi tọa độ Bounding Box.
2. Tính cân bằng nhãn (Class Symmetry & Balance): Thống kê số lượng Bounding Box theo từng lớp Fresh/Rotten.
3. Chống rò rỉ dữ liệu (Data Leakage Prevention): Kiểm tra mã băm (MD5 hash) giữa tập Train / Valid / Test.
4. Xuất biểu đồ phân bố nhãn và báo cáo Ma trận đánh giá chất lượng dữ liệu.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from config import CLASS_NAMES, REPORTS_DIR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def compute_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_yolo_label_file(lbl_path: Path, num_classes: int):
    """
    Kiểm tra một file nhãn YOLO (.txt).
    Trả về (is_valid, reason, list_of_class_ids)
    """
    if not lbl_path.exists():
        return False, "missing_label_file", []

    content = lbl_path.read_text(encoding="utf-8").strip()
    if not content:
        return False, "empty_label_file", []

    class_ids = []
    for line_idx, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            return False, f"invalid_format_line_{line_idx}", []
        try:
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            return False, f"non_numeric_line_{line_idx}", []

        if not (0 <= cls_id < num_classes):
            return False, f"class_id_out_of_range_{cls_id}", []
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            return False, f"bbox_out_of_bounds_line_{line_idx}", []

        class_ids.append(cls_id)

    return True, "valid", class_ids


def run_data_quality_audit(data_yaml_path: Path, clean_invalid: bool = True) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get("path", data_yaml_path.parent))
    class_names = data_cfg.get("names", CLASS_NAMES)
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names.keys())]
    num_classes = len(class_names)

    splits = {"train": "train", "valid": "valid", "test": "test"}
    split_hashes = {s: set() for s in splits}
    class_counts_per_split = {s: Counter() for s in splits}
    total_class_counts = Counter()
    removed_files = []
    valid_images_per_split = {}

    print("=" * 70)
    print("BẮT ĐẦU KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY AUDIT - TUẦN 3)")
    print("=" * 70)

    for split_name in splits:
        split_dir = base_path / split_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists():
            continue

        valid_img_count = 0
        image_files = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        )

        for img_path in image_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            is_valid, reason, cls_ids = validate_yolo_label_file(lbl_path, num_classes)

            if not is_valid:
                removed_files.append(
                    {
                        "split": split_name,
                        "image": img_path.name,
                        "label": lbl_path.name,
                        "reason": reason,
                    }
                )
                if clean_invalid:
                    if img_path.exists():
                        img_path.unlink()
                    if lbl_path.exists():
                        lbl_path.unlink()
                continue

            valid_img_count += 1
            split_hashes[split_name].add(compute_md5(img_path))
            for cid in cls_ids:
                class_counts_per_split[split_name][cid] += 1
                total_class_counts[cid] += 1

        valid_images_per_split[split_name] = valid_img_count

    # Kiểm tra Data Leakage giữa Train / Valid / Test
    leak_train_val = len(split_hashes["train"].intersection(split_hashes["valid"]))
    leak_train_test = len(split_hashes["train"].intersection(split_hashes["test"]))
    leak_val_test = len(split_hashes["valid"].intersection(split_hashes["test"]))
    total_leaks = leak_train_val + leak_train_test + leak_val_test

    # Tính toán cân bằng nhóm Fresh vs Rotten
    fresh_total = sum(
        total_class_counts[i] for i, name in enumerate(class_names) if "fresh" in name.lower()
    )
    rotten_total = sum(
        total_class_counts[i] for i, name in enumerate(class_names) if "rotten" in name.lower()
    )
    balance_ratio = round(fresh_total / max(1, rotten_total), 3)

    # Vẽ biểu đồ phân bố Bounding Box theo từng lớp (EDA & Audit Chart)
    labels_display = [class_names[i] for i in range(num_classes)]
    counts_display = [total_class_counts[i] for i in range(num_classes)]
    colors = ["#2ecc71" if "fresh" in name.lower() else "#e74c3c" for name in labels_display]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    bars = ax.bar(labels_display, counts_display, color=colors, edgecolor="#2c3e50", width=0.6)
    ax.set_title("Phân bố số lượng Bounding Box theo Lớp (Fresh vs Rotten)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Số lượng Bounding Box", fontsize=11)
    ax.set_xlabel("Các lớp Nông sản (Táo, Chuối, Ớt chuông)", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + max(1, max(counts_display) * 0.01), f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    chart_path = REPORTS_DIR / "class_distribution.png"
    plt.savefig(chart_path)
    plt.close(fig)

    audit_summary = {
        "dataset_yaml": str(data_yaml_path),
        "valid_images_per_split": valid_images_per_split,
        "total_valid_images": sum(valid_images_per_split.values()),
        "total_bounding_boxes": sum(total_class_counts.values()),
        "class_distribution": {class_names[i]: total_class_counts[i] for i in range(num_classes)},
        "fresh_vs_rotten": {
            "fresh_boxes": fresh_total,
            "rotten_boxes": rotten_total,
            "fresh_to_rotten_ratio": balance_ratio,
        },
        "data_leakage_check": {
            "train_val_overlap": leak_train_val,
            "train_test_overlap": leak_train_test,
            "val_test_overlap": leak_val_test,
            "status": "PASSED (0 rò rỉ dữ liệu)" if total_leaks == 0 else f"WARNING ({total_leaks} ảnh trùng)",
        },
        "cleaned_noisy_files_count": len(removed_files),
        "cleaned_noisy_files": removed_files,
        "chart_path": str(chart_path),
    }

    audit_json_path = REPORTS_DIR / "data_quality_audit.json"
    audit_json_path.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"- Tổng số ảnh hợp lệ sau làm sạch: {audit_summary['total_valid_images']} ảnh {valid_images_per_split}")
    print(f"- Số file nhãn lỗi/rỗng đã tự động phát hiện & loại bỏ: {len(removed_files)} file")
    for rf in removed_files:
        print(f"   + [{rf['split']}] {rf['label']} -> Lý do: {rf['reason']}")
    print(f"- Tổng số Bounding Box: {audit_summary['total_bounding_boxes']} (Fresh: {fresh_total}, Rotten: {rotten_total}, Tỷ lệ: {balance_ratio})")
    print(f"- Kiểm tra Data Leakage (MD5 Hash): {audit_summary['data_leakage_check']['status']}")
    print(f"- Đã lưu biểu đồ phân bố nhãn tại: {chart_path}")
    print(f"- Đã lưu báo cáo JSON tại: {audit_json_path}")
    return audit_summary


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parent / "dataset" / "data.yaml"
    parser = argparse.ArgumentParser(description="Kiểm định chất lượng dữ liệu YOLOv11 (Data Quality Audit)")
    parser.add_argument("--data", type=str, default=str(default_data), help="Đường dẫn tới file data.yaml")
    args = parser.parse_args()

    run_data_quality_audit(Path(args.data), clean_invalid=True)
