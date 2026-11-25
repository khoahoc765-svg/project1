"""
Cấu hình trung tâm cho Hệ thống Phát hiện và Phân loại Độ tươi Nông sản (YOLOv11n)
Đề tài: Automated Food & Produce Freshness Detection System
"""

import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
for _venv in (BASE_DIR / ".venv", BASE_DIR.parent.parent / ".venv"):
    if (_venv / "bin" / "python").exists():
        if Path(sys.prefix).resolve() != _venv.resolve():
            os.execv(str(_venv / "bin" / "python"), [str(_venv / "bin" / "python"), sys.argv[0]] + sys.argv[1:])
        break

CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))

DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"
RUNS_DIR = BASE_DIR / "runs"

# Thông tin Dataset Roboflow theo Báo cáo Tuần 2 (Mục 2.1)
ROBOFLOW_WORKSPACE = "freshness-detection-vwn1a"
ROBOFLOW_PROJECT = "freshness-detection-rhrze"
ROBOFLOW_VERSION = 1

# 6 nhãn phân loại trọng tâm (3 loại quả đại diện: Táo, Chuối, Ớt chuông x 2 trạng thái Fresh/Rotten)
CLASS_NAMES = [
    "Fresh apple",
    "Fresh banana",
    "Fresh bell pepper",
    "Rotten apple",
    "Rotten banana",
    "Rotten bell pepper",
]

# Mô hình cơ sở (Mục 2.2: YOLOv11 phiên bản nano siêu nhẹ)
MODEL_WEIGHTS = str(BASE_DIR / "yolo11n.pt")
IMG_SIZE = 640

# Cấu hình Baseline Model - Tuần 3 (Mục 3.4)
BASELINE_CONFIG = {
    "epochs": 15,
    "imgsz": IMG_SIZE,
    "batch": 16,
    "project": str(RUNS_DIR),
    "name": "week3_baseline_yolo11n",
    "exist_ok": True,
    "pretrained": True,
    "verbose": True,
}

# Cấu hình Tuned Model & Controlled Augmentation - Tuần 3 & Tuần 4 (Mục 2.3, 3.2 & 3.4)
TUNED_WEEK4_CONFIG = {
    "epochs": 50,
    "imgsz": IMG_SIZE,
    "batch": 16,
    "project": str(RUNS_DIR),
    "name": "week4_tuned_yolo11n",
    "exist_ok": True,
    "pretrained": True,
    # Tinh chỉnh Hyperparameter (Tuần 4)
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "weight_decay": 0.0005,
    # Chiến lược Data Augmentation có kiểm soát (Mục 3.2)
    "hsv_h": 0.005,   # Cố định tông màu để bảo toàn màu sắc thật của vết thối/mốc
    "hsv_s": 0.5,     # Điều chỉnh độ bão hòa vừa phải
    "hsv_v": 0.3,     # Điều chỉnh độ sáng mô phỏng ánh đèn kho bảo quản
    "mosaic": 1.0,    # Bật Mosaic = 1.0 khắc phục Lab Bias & mô phỏng quả xếp chồng che khuất
    "degrees": 10.0,  # Xoay nhẹ mô phỏng góc Camera trên băng chuyền
    "fliplr": 0.5,    # Lật ngang tự nhiên
    "flipud": 0.1,    # Lật dọc nhẹ
    "scale": 0.4,     # Co giãn tỷ lệ khoảng cách camera
    "verbose": True,
}
