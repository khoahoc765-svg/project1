"""
Bước 5: Đánh giá Mô hình (Confusion Matrix, mAP) & Dự đoán thực tế (Inference - Tuần 4, Mục 3.4)
"""

import argparse
import json
import shutil
from pathlib import Path
from config import CLASS_NAMES, IMG_SIZE, REPORTS_DIR, RUNS_DIR
from ultralytics import YOLO

from step3_train_baseline import get_device


def evaluate_and_run_inference(
    weights_path: Path,
    data_yaml_path: Path,
    source_dir: Path = None,
    imgsz: int = IMG_SIZE,
    conf_thres: float = 0.25,
) -> dict:
    device = get_device()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    inference_out_dir = REPORTS_DIR / "inference_results"
    inference_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ĐÁNH GIÁ MÔ HÌNH & PHÂN TÍCH CONFUSION MATRIX | Trọng số: {weights_path}")
    print("=" * 70)

    model = YOLO(str(weights_path))
    val_metrics = model.val(
        data=str(data_yaml_path),
        split="test",
        imgsz=imgsz,
        device=device,
        project=str(RUNS_DIR),
        name="week4_evaluation",
        exist_ok=True,
        plots=True,
    )

    val_save_dir = Path(val_metrics.save_dir)
    # Sao chép Confusion Matrix & Curves sang thư mục reports để tiện đưa vào báo cáo
    for cm_name in ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png", "PR_curve.png", "F1_curve.png"):
        src_cm = val_save_dir / cm_name
        if src_cm.exists():
            shutil.copy2(src_cm, REPORTS_DIR / cm_name)

    # Thu thập chỉ số đánh giá theo từng lớp và toàn cục
    names_map = val_metrics.names if hasattr(val_metrics, "names") else {i: n for i, n in enumerate(CLASS_NAMES)}
    per_class_metrics = {}
    if hasattr(val_metrics, "box") and hasattr(val_metrics.box, "ap_class_index"):
        for idx_pos, cls_idx in enumerate(val_metrics.box.ap_class_index):
            cls_name = names_map.get(int(cls_idx), str(cls_idx))
            p_i, r_i, ap50_i, ap_i = val_metrics.box.class_result(idx_pos)
            per_class_metrics[cls_name] = {
                "precision": round(float(p_i), 4),
                "recall": round(float(r_i), 4),
                "mAP50": round(float(ap50_i), 4),
                "mAP50_95": round(float(ap_i), 4),
            }

    summary_metrics = {
        "weights_path": str(weights_path),
        "overall": {
            "precision": round(float(val_metrics.box.mp), 4),
            "recall": round(float(val_metrics.box.mr), 4),
            "mAP50": round(float(val_metrics.box.map50), 4),
            "mAP50_95": round(float(val_metrics.box.map), 4),
        },
        "per_class": per_class_metrics,
        "confusion_matrix_path": str(REPORTS_DIR / "confusion_matrix.png"),
        "confusion_matrix_normalized_path": str(REPORTS_DIR / "confusion_matrix_normalized.png"),
    }

    # Chạy Inference (Dự đoán thử nghiệm trên tập ảnh thực tế/test chưa từng gặp)
    if source_dir is None:
        candidate = data_yaml_path.parent / "test" / "images"
        if candidate.exists():
            source_dir = candidate
        else:
            source_dir = Path(__file__).resolve().parent / "datasets" / "freshness_yolov11_dataset" / "test" / "images"

    print("=" * 70)
    print(f"CHẠY DỰ ĐOÁN THỰC TẾ (INFERENCE) TRÊN: {source_dir}")
    print("=" * 70)

    preds = model.predict(
        source=str(source_dir),
        imgsz=imgsz,
        conf=conf_thres,
        device=device,
        save=True,
        project=str(REPORTS_DIR),
        name="inference_results",
        exist_ok=True,
    )

    detections_summary = []
    for pred in preds[:10]:  # Lưu chi tiết 10 ảnh đầu tiên vào báo cáo JSON
        img_name = Path(pred.path).name
        boxes_info = []
        if pred.boxes is not None:
            for b in pred.boxes:
                cid = int(b.cls.item())
                conf = round(float(b.conf.item()), 4)
                xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]
                boxes_info.append(
                    {
                        "class_id": cid,
                        "class_name": names_map.get(cid, str(cid)),
                        "confidence": conf,
                        "bbox_xyxy": xyxy,
                    }
                )
        detections_summary.append({"image": img_name, "detections": boxes_info})

    summary_metrics["sample_inference_predictions"] = detections_summary
    summary_metrics["inference_output_dir"] = str(inference_out_dir)

    metrics_json_path = REPORTS_DIR / "evaluation_metrics.json"
    metrics_json_path.write_text(json.dumps(summary_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"- Kết quả Tổng thể trên tập Test: Precision={summary_metrics['overall']['precision']:.4f}, "
          f"Recall={summary_metrics['overall']['recall']:.4f}, "
          f"mAP@50={summary_metrics['overall']['mAP50']:.4f}, "
          f"mAP@50-95={summary_metrics['overall']['mAP50_95']:.4f}")
    print(f"- Đã xuất Confusion Matrix tại: {REPORTS_DIR / 'confusion_matrix.png'}")
    print(f"- Đã lưu ảnh dự đoán Bounding Box tại: {inference_out_dir}")
    return summary_metrics


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    default_weights = base_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    default_data = base_dir / "dataset" / "data.yaml"
    parser = argparse.ArgumentParser(description="Đánh giá Confusion Matrix & Chạy Inference YOLOv11n (Tuần 4)")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Đường dẫn tới file trọng số best.pt")
    parser.add_argument("--data", type=str, default=str(default_data), help="Đường dẫn tới file data.yaml")
    parser.add_argument("--source", type=str, default=None, help="Thư mục ảnh/video hoặc đường dẫn file cần dự đoán")
    parser.add_argument("--imgsz", type=int, default=256, help="Kích thước ảnh chuẩn hóa")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng độ tin cậy (Confidence Threshold)")
    args = parser.parse_args()

    src_path = Path(args.source) if args.source else None
    evaluate_and_run_inference(
        weights_path=Path(args.weights),
        data_yaml_path=Path(args.data),
        source_dir=src_path,
        imgsz=args.imgsz,
        conf_thres=args.conf,
    )
