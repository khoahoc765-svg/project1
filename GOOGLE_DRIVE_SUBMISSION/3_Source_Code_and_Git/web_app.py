"""
Giao diện Web Nhận diện Độ tươi Nông sản (YOLOv11n + HSV Controlled Analysis)
Chạy cục bộ tại: http://localhost:8085
"""

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
for _venv in (BASE_DIR / ".venv", BASE_DIR.parent.parent / ".venv"):
    if (_venv / "bin" / "python").exists():
        if Path(sys.prefix).resolve() != _venv.resolve():
            os.execv(str(_venv / "bin" / "python"), [str(_venv / "bin" / "python"), __file__] + sys.argv[1:])
        break

PROJECT_ROOT = BASE_DIR if (BASE_DIR / "runs").exists() else BASE_DIR.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))

import cv2
import numpy as np
from ultralytics import YOLO

from detect import CLASS_NAMES, run_inference_on_image

WEIGHTS_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "freshness_yolov11_dataset" / "test" / "images"
TRAIN_RUN_DIR = PROJECT_ROOT / "runs" / "detect" / "train"

MODEL = YOLO(str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else "yolo11n.yaml")


def encode_bgr_to_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def decode_base64_to_bgr(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>YOLOv11 Freshness Detection - Hệ thống Nhận diện Độ tươi Nông sản</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --fresh: #22c55e;
      --rotten: #ef4444;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1280px; margin: 0 auto; }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    h1 { font-size: 1.5rem; font-weight: 700; color: #fff; }
    .subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 4px; }
    .badges { display: flex; gap: 8px; flex-wrap: wrap; }
    .badge {
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: var(--accent);
      font-size: 0.78rem;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 600;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 24px;
    }
    @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
    }
    .card h2 {
      font-size: 1.05rem;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .controls { display: flex; flex-direction: column; gap: 16px; }
    .btn {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 10px 16px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.9rem;
      transition: opacity 0.15s;
      text-align: center;
    }
    .btn:hover { opacity: 0.9; }
    .btn-outline {
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border);
    }
    .btn-outline:hover { background: rgba(255,255,255,0.05); }
    .sample-list {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      max-height: 220px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .sample-btn {
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 7px 6px;
      border-radius: 6px;
      font-size: 0.78rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .sample-btn:hover, .sample-btn.active {
      border-color: var(--accent);
      color: #fff;
      background: rgba(56, 189, 248, 0.15);
    }
    .slider-row { display: flex; flex-direction: column; gap: 6px; }
    .slider-row label { font-size: 0.85rem; color: var(--muted); display: flex; justify-content: space-between; }
    .viewer-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    @media (max-width: 700px) { .viewer-grid { grid-template-columns: 1fr; } }
    .img-box {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      text-align: center;
      min-height: 280px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    .img-box span { font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }
    .img-box img, .img-box video {
      max-width: 100%;
      max-height: 340px;
      border-radius: 8px;
      object-fit: contain;
    }
    .stats-bar {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 18px;
    }
    .stat-item {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      text-align: center;
    }
    .stat-val { font-size: 1.5rem; font-weight: 700; }
    .stat-label { font-size: 0.78rem; color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--muted); font-weight: 600; }
    .pill-fresh {
      background: rgba(34, 197, 94, 0.15);
      color: var(--fresh);
      padding: 3px 9px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 0.78rem;
    }
    .pill-rotten {
      background: rgba(239, 68, 68, 0.15);
      color: var(--rotten);
      padding: 3px 9px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 0.78rem;
    }
    input[type="file"] { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Hệ thống Nhận diện Độ tươi Nông sản (YOLOv11n)</h1>
        <div class="subtitle">Trọng số: <code>runs/detect/train/weights/best.pt</code> | 6 lớp: Táo, Chuối, Ớt chuông (Fresh / Rotten)</div>
      </div>
      <div class="badges">
        <span class="badge">YOLOv11n</span>
        <span class="badge">mosaic = 1.0</span>
        <span class="badge">hsv_h = 0.005</span>
        <span class="badge">hsv_s = 0.3</span>
        <span class="badge">hsv_v = 0.3</span>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <h2>Bảng Điều khiển Dự đoán</h2>
        <div class="controls">
          <label class="btn" for="fileInput">Tải ảnh nông sản từ máy tính</label>
          <input type="file" id="fileInput" accept="image/*" />

          <button class="btn btn-outline" id="webcamBtn" onclick="toggleWebcam()">Bật Camera / Webcam trực tiếp</button>

          <div class="slider-row">
            <label>
              <span>Ngưỡng độ tin cậy (Confidence)</span>
              <strong id="confVal">0.25</strong>
            </label>
            <input type="range" id="confSlider" min="0.05" max="0.90" step="0.05" value="0.25" onchange="updateConf(this.value)" />
          </div>

          <div>
            <div style="font-size: 0.85rem; color: var(--muted); margin-bottom: 8px;">
              Chọn nhanh ảnh mẫu từ tập <code>test/images</code>:
            </div>
            <div class="sample-list" id="sampleList"></div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Kết quả Nhận diện & Phân tích HSV <span id="statusText" style="font-size:0.8rem;color:var(--accent);">Sẵn sàng</span></h2>

        <div class="stats-bar">
          <div class="stat-item">
            <div class="stat-val" id="totalCount">0</div>
            <div class="stat-label">Tổng số nông sản</div>
          </div>
          <div class="stat-item">
            <div class="stat-val" style="color: var(--fresh);" id="freshCount">0</div>
            <div class="stat-label">Tươi (Đạt chuẩn)</div>
          </div>
          <div class="stat-item">
            <div class="stat-val" style="color: var(--rotten);" id="rottenCount">0</div>
            <div class="stat-label">Hỏng (Cần loại bỏ)</div>
          </div>
        </div>

        <div class="viewer-grid">
          <div class="img-box">
            <span>Ảnh đầu vào / Camera</span>
            <img id="inputImg" alt="Ảnh gốc" />
            <video id="webcamVideo" autoplay playsinline style="display:none;"></video>
          </div>
          <div class="img-box">
            <span>Kết quả Bounding Box (YOLOv11)</span>
            <img id="outputImg" alt="Ảnh dự đoán" />
          </div>
        </div>

        <h2 style="margin-top: 8px;">Chi tiết Đối tượng Phát hiện</h2>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Phân loại (Class)</th>
                <th>Trạng thái</th>
                <th>Độ tin cậy</th>
                <th>Đặc trưng màu HSV & Đốm thâm</th>
                <th>Tọa độ Box [x1, y1, x2, y2]</th>
              </tr>
            </thead>
            <tbody id="detTableBody">
              <tr><td colspan="6" style="color:var(--muted);text-align:center;">Chọn một ảnh mẫu bên trái hoặc tải ảnh lên để xem kết quả.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentSample = null;
    let currentDataUrl = null;
    let webcamStream = null;
    let webcamTimer = null;

    async function loadSamples() {
      const res = await fetch('/api/samples');
      const data = await res.json();
      const container = document.getElementById('sampleList');
      container.innerHTML = '';
      data.samples.forEach((name, idx) => {
        const btn = document.createElement('button');
        btn.className = 'sample-btn';
        btn.textContent = name.replace('test_produce_', 'Mẫu #').replace('.jpg', '');
        btn.onclick = () => selectSample(name, btn);
        container.appendChild(btn);
        if (idx === 0) selectSample(name, btn);
      });
    }

    function updateConf(val) {
      document.getElementById('confVal').textContent = Number(val).toFixed(2);
      if (currentDataUrl) {
        runPredictDataUrl(currentDataUrl);
      } else if (currentSample) {
        runPredictSample(currentSample);
      }
    }

    async function selectSample(name, btnEl) {
      stopWebcam();
      currentSample = name;
      currentDataUrl = null;
      document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      if (btnEl) btnEl.classList.add('active');
      await runPredictSample(name);
    }

    async function runPredictSample(name) {
      const conf = document.getElementById('confSlider').value;
      document.getElementById('statusText').textContent = 'Đang phân tích...';
      const res = await fetch(`/api/predict_sample?name=${encodeURIComponent(name)}&conf=${conf}`);
      const data = await res.json();
      renderResult(data);
    }

    async function runPredictDataUrl(dataUrl) {
      const conf = document.getElementById('confSlider').value;
      document.getElementById('statusText').textContent = 'Đang phân tích...';
      const res = await fetch('/api/predict_upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl, conf: parseFloat(conf) })
      });
      const data = await res.json();
      renderResult(data);
    }

    function renderResult(data) {
      document.getElementById('statusText').textContent = 'Hoàn tất';
      if (data.input_image) {
        const inImg = document.getElementById('inputImg');
        inImg.style.display = 'block';
        inImg.src = data.input_image;
      }
      if (data.output_image) {
        document.getElementById('outputImg').src = data.output_image;
      }
      const dets = data.detections || [];
      const fresh = dets.filter(d => d.is_fresh).length;
      const rotten = dets.length - fresh;
      document.getElementById('totalCount').textContent = dets.length;
      document.getElementById('freshCount').textContent = fresh;
      document.getElementById('rottenCount').textContent = rotten;

      const tbody = document.getElementById('detTableBody');
      if (dets.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;">Không phát hiện nông sản nào vượt ngưỡng tin cậy.</td></tr>';
        return;
      }
      tbody.innerHTML = dets.map((d, i) => `
        <tr>
          <td>${i + 1}</td>
          <td><strong>${d.class_vi}</strong></td>
          <td><span class="${d.is_fresh ? 'pill-fresh' : 'pill-rotten'}">${d.is_fresh ? 'Tươi (Đạt)' : 'Hỏng (Loại)'}</span></td>
          <td>${(d.confidence * 100).toFixed(1)}%</td>
          <td>H:${d.hsv_stats.h} | S:${d.hsv_stats.s} | V:${d.hsv_stats.v} (Thâm: ${d.hsv_stats.dark_spot_ratio}%)</td>
          <td><code>[${d.bbox.join(', ')}]</code></td>
        </tr>
      `).join('');
    }

    document.getElementById('fileInput').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      stopWebcam();
      const reader = new FileReader();
      reader.onload = (ev) => {
        currentDataUrl = ev.target.result;
        currentSample = null;
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
        runPredictDataUrl(currentDataUrl);
      };
      reader.readAsDataURL(file);
    });

    async function toggleWebcam() {
      if (webcamStream) {
        stopWebcam();
        return;
      }
      try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
        const video = document.getElementById('webcamVideo');
        document.getElementById('inputImg').style.display = 'none';
        video.style.display = 'block';
        video.srcObject = webcamStream;
        document.getElementById('webcamBtn').textContent = 'Dừng Camera / Webcam';
        const canvas = document.createElement('canvas');
        webcamTimer = setInterval(async () => {
          if (!webcamStream || video.videoWidth === 0) return;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          const frameUrl = canvas.toDataURL('image/jpeg', 0.8);
          const conf = document.getElementById('confSlider').value;
          const res = await fetch('/api/predict_upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frameUrl, conf: parseFloat(conf) })
          });
          const data = await res.json();
          data.input_image = null;
          renderResult(data);
        }, 700);
      } catch (err) {
        alert('Không thể truy cập Camera: ' + err.message);
      }
    }

    function stopWebcam() {
      if (webcamTimer) { clearInterval(webcamTimer); webcamTimer = null; }
      if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
      }
      document.getElementById('webcamVideo').style.display = 'none';
      document.getElementById('inputImg').style.display = 'block';
      document.getElementById('webcamBtn').textContent = 'Bật Camera / Webcam trực tiếp';
    }

    loadSamples();
  </script>
</body>
</html>
"""


class FreshnessWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/samples":
            samples = []
            if TEST_IMAGES_DIR.exists():
                samples = sorted([p.name for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() == ".jpg"])
            self._send_json({"samples": samples})
            return

        if parsed.path == "/api/predict_sample":
            qs = parse_qs(parsed.query)
            name = qs.get("name", [""])[0]
            conf = float(qs.get("conf", ["0.25"])[0])
            img_path = TEST_IMAGES_DIR / Path(name).name
            if not img_path.exists():
                self._send_json({"error": "Không tìm thấy ảnh mẫu"}, status=404)
                return
            img_bgr = cv2.imread(str(img_path))
            dets, annotated = run_inference_on_image(MODEL, img_bgr, conf=conf)
            self._send_json({
                "input_image": encode_bgr_to_base64(img_bgr),
                "output_image": encode_bgr_to_base64(annotated),
                "detections": dets,
            })
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict_upload":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            conf = float(payload.get("conf", 0.25))
            img_bgr = decode_base64_to_bgr(payload.get("image", ""))
            if img_bgr is None:
                self._send_json({"error": "Ảnh không hợp lệ"}, status=400)
                return
            dets, annotated = run_inference_on_image(MODEL, img_bgr, conf=conf)
            self._send_json({
                "input_image": encode_bgr_to_base64(img_bgr),
                "output_image": encode_bgr_to_base64(annotated),
                "detections": dets,
            })
            return

        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy Web App Nhận diện Độ tươi Nông sản YOLOv11")
    parser.add_argument("--port", type=int, default=8085, help="Cổng chạy Web Server (mặc định: 8085)")
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), FreshnessWebHandler)
    print("=" * 68)
    print(f"Web App Nhận diện Độ tươi Nông sản đã khởi chạy tại: http://localhost:{args.port}")
    print(f"Đang sử dụng trọng số: {WEIGHTS_PATH}")
    print("=" * 68)
    server.serve_forever()


if __name__ == "__main__":
    main()
