"""Flask web app: stream annotated webcam over HTTP and serve Wikipedia
details on demand. Run with: python -m app.web [--source X --port 5000 ...]"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

from .detector import Detector
from .info_fetcher import fetch_details
from .ui import draw_boxes


class CameraThread(threading.Thread):
    """One background thread reads frames so multiple HTTP clients can share."""

    def __init__(self, source, detector, jpeg_quality: int = 80):
        super().__init__(daemon=True)
        self.source = source
        self.detector = detector
        self.jpeg_quality = jpeg_quality
        self.frame_lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_detections: list = []
        self.stop_event = threading.Event()
        self.error: str | None = None

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"Could not open source {self.source!r}"
            print(f"[web] {self.error}")
            return
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                dets = self.detector.detect(frame)
                draw_boxes(frame, dets, selected_idx=-1)
                ret, jpg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ret:
                    continue
                with self.frame_lock:
                    self.latest_jpeg = jpg.tobytes()
                    self.latest_detections = dets
        finally:
            cap.release()

    def get_jpeg(self) -> bytes | None:
        with self.frame_lock:
            return self.latest_jpeg

    def get_detections(self) -> list:
        with self.frame_lock:
            return list(self.latest_detections)


def _resolve_source(src: str):
    return int(src) if src.isdigit() else src


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web UI for the object detector.")
    p.add_argument("--source", default="0", help="Webcam index or stream URL.")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 = any device on LAN).")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-track", action="store_true")
    return p.parse_args()


def create_app(camera: CameraThread) -> Flask:
    base = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(base / "templates"),
        static_folder=str(base / "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/stream")
    def stream():
        def generator():
            while True:
                if camera.error:
                    break
                jpg = camera.get_jpeg()
                if jpg is None:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )
                time.sleep(0.033)  # cap at ~30 fps over the wire

        return Response(generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/detections")
    def api_detections():
        dets = camera.get_detections()
        return jsonify(
            [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "box": list(d.box),
                    "track_id": d.track_id,
                }
                for d in dets
            ]
        )

    @app.route("/api/details")
    def api_details():
        label = request.args.get("label", "").strip()
        lang = request.args.get("lang", "en").strip() or "en"
        if not label:
            return jsonify({"error": "missing label"}), 400
        return jsonify(fetch_details(label, lang=lang))

    @app.route("/api/health")
    def api_health():
        return jsonify({"ok": camera.error is None, "error": camera.error})

    return app


def main() -> int:
    args = parse_args()
    detector = Detector(weights=args.weights, conf=args.conf, track=not args.no_track)
    camera = CameraThread(source=_resolve_source(args.source), detector=detector)
    camera.start()

    app = create_app(camera)
    print(f"[web] open http://{args.host}:{args.port}/  (also reachable from phones on same Wi-Fi)")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        camera.stop_event.set()
        camera.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
