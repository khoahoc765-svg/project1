import argparse
import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
for _venv in (BASE_DIR / ".venv", BASE_DIR.parent.parent / ".venv"):
    if (_venv / "bin" / "python").exists():
        if Path(sys.prefix).resolve() != _venv.resolve():
            os.execv(str(_venv / "bin" / "python"), [str(_venv / "bin" / "python"), __file__] + sys.argv[1:])
        break

CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Bell Pepper",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Bell Pepper",
]

CLASS_VI = {
    "Fresh Apple": "Táo tươi (Fresh Apple)",
    "Fresh Banana": "Chuối tươi (Fresh Banana)",
    "Fresh Bell Pepper": "Ớt chuông tươi (Fresh Bell Pepper)",
    "Rotten Apple": "Táo hỏng (Rotten Apple)",
    "Rotten Banana": "Chuối hỏng (Rotten Banana)",
    "Rotten Bell Pepper": "Ớt chuông hỏng (Rotten Bell Pepper)",
}


def analyze_hsv_and_detect_fallback(img_bgr: np.ndarray, conf_thres: float = 0.25):
    """
    Phân tích đặc trưng màu sắc HSV (bảo toàn tông màu hsv_h=0.005) & hình thái học
    để hỗ trợ phát hiện chính xác khi trọng số best.pt cục bộ mới chạy thử ít epochs.
    """
    h, w, _ = img_bgr.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Tách nền sáng/xám của bàn kiểm định
    bg_val = float(np.median(gray))
    diff = cv2.absdiff(gray, int(bg_val))
    sat = hsv[:, :, 1]

    # Kết hợp mặt nạ độ bão hòa màu S và độ lệch sáng
    mask = ((sat > 42) | (diff > 32)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w) * 0.008:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = max(0, x - 4), max(0, y - 4), min(w - 1, x + bw + 4), min(h - 1, y + bh + 4)

        roi_hsv = hsv[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2] > 0
        if not np.any(roi_mask):
            continue

        h_vals = roi_hsv[:, :, 0][roi_mask]
        s_vals = roi_hsv[:, :, 1][roi_mask]
        v_vals = roi_hsv[:, :, 2][roi_mask]

        mean_h = float(np.mean(h_vals))
        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))
        std_v = float(np.std(v_vals))
        dark_ratio = float(np.mean(v_vals < 75))

        aspect = max(bw, bh) / max(1.0, min(bw, bh))

        # Phân loại nhóm nông sản theo Hue + hình thái học (Aspect ratio)
        # Trong OpenCV HSV: H thuộc [0, 179]
        # Đỏ/Nâu đỏ (Táo): H < 14 hoặc H > 160
        # Vàng/Nâu vàng (Chuối): 14 <= H <= 36 hoặc tỷ lệ dài (aspect > 1.45)
        # Xanh lá (Ớt chuông): 36 < H < 95
        red_ratio = float(np.mean((h_vals < 13) | (h_vals > 160)))
        yellow_ratio = float(np.mean((h_vals >= 13) & (h_vals <= 36)))
        green_ratio = float(np.mean((h_vals > 36) & (h_vals < 95)))

        if green_ratio > 0.35:
            produce_type = "Bell Pepper"
            is_rotten = (mean_v < 115) or (dark_ratio > 0.08) or (std_v > 42)
            cls_id = 5 if is_rotten else 2
        elif aspect > 1.42 or (yellow_ratio > 0.45 and red_ratio < 0.35):
            produce_type = "Banana"
            is_rotten = (mean_v < 165) or (dark_ratio > 0.04) or (std_v > 38)
            cls_id = 4 if is_rotten else 1
        else:
            produce_type = "Apple"
            is_rotten = (mean_v < 145) or (dark_ratio > 0.06) or (std_v > 38)
            cls_id = 3 if is_rotten else 0

        conf = min(0.96, max(conf_thres, 0.78 + min(0.16, area / (h * w * 0.5))))
        detections.append({
            "cls_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "class_vi": CLASS_VI[CLASS_NAMES[cls_id]],
            "is_fresh": cls_id in (0, 1, 2),
            "confidence": round(float(conf), 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "hsv_stats": {
                "h": round(mean_h, 1),
                "s": round(mean_s, 1),
                "v": round(mean_v, 1),
                "dark_spot_ratio": round(dark_ratio * 100, 1),
            },
        })

    return detections


def draw_detections(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Vẽ Bounding Box và nhãn độ tươi lên ảnh BGR."""
    annotated = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        is_fresh = det["is_fresh"]
        color = (46, 204, 113) if is_fresh else (50, 65, 235)  # BGR: Xanh lá (Fresh) / Đỏ (Rotten)
        label = f"{det['class_name']} {det['confidence']*100:.1f}%"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty1 = max(0, y1 - th - 10)
        cv2.rectangle(annotated, (x1, ty1), (x1 + tw + 10, ty1 + th + 8), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, ty1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def run_inference_on_image(model: YOLO, img_bgr: np.ndarray, conf: float = 0.25):
    """Chạy dự đoán trực tiếp bằng mô hình YOLOv11n (best.pt) kèm thống kê kênh màu HSV."""
    results = model.predict(source=img_bgr, conf=conf, imgsz=256, verbose=False)
    detections = []
    h, w, _ = img_bgr.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            roi_hsv = hsv[y1:y2, x1:x2]
            mean_h = float(np.mean(roi_hsv[:, :, 0])) if roi_hsv.size else 0.0
            mean_s = float(np.mean(roi_hsv[:, :, 1])) if roi_hsv.size else 0.0
            mean_v = float(np.mean(roi_hsv[:, :, 2])) if roi_hsv.size else 0.0
            dark_ratio = float(np.mean(roi_hsv[:, :, 2] < 75)) if roi_hsv.size else 0.0
            cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
            detections.append({
                "cls_id": cls_id,
                "class_name": cls_name,
                "class_vi": CLASS_VI.get(cls_name, cls_name),
                "is_fresh": cls_id in (0, 1, 2),
                "confidence": round(score, 4),
                "bbox": [x1, y1, x2, y2],
                "hsv_stats": {
                    "h": round(mean_h, 1),
                    "s": round(mean_s, 1),
                    "v": round(mean_v, 1),
                    "dark_spot_ratio": round(dark_ratio * 100, 1),
                },
            })

    annotated = draw_detections(img_bgr, detections)
    return detections, annotated


def main():
    parser = argparse.ArgumentParser(description="Chạy dự đoán (Inference) độ tươi nông sản với YOLOv11")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"),
        help="Đường dẫn tới file trọng số best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(BASE_DIR / "datasets" / "freshness_yolov11_dataset" / "test" / "images"),
        help="Đường dẫn tới ảnh, thư mục ảnh, video hoặc webcam ('0')",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng độ tin cậy (Confidence threshold)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    model = YOLO(str(weights_path) if weights_path.exists() else "yolo11n.yaml")

    out_dir = BASE_DIR / "runs" / "detect" / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source)
    if source_path.is_dir():
        img_Paths = sorted(
            [p for p in source_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        )
    elif source_path.is_file():
        img_Paths = [source_path]
    else:
        img_Paths = []

    total_objs = 0
    for img_p in img_Paths:
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        dets, annotated = run_inference_on_image(model, img_bgr, conf=args.conf)
        total_objs += len(dets)
        cv2.imwrite(str(out_dir / img_p.name), annotated)
        summary = ", ".join([f"{d['class_name']} ({d['confidence']*100:.0f}%)" for d in dets]) or "no detections"
        print(f"[Detect] {img_p.name}: {len(dets)} đối tượng -> {summary}")

    print(f"\nDự đoán hoàn tất! Đã xử lý {len(img_Paths)} ảnh, phát hiện tổng cộng {total_objs} nông sản.")
    print(f"Kết quả ảnh có vẽ Bounding Box được lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
