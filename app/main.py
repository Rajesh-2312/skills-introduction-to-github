import argparse
import time
from collections import deque
from pathlib import Path

import cv2

from .detector import Detector
from .info_fetcher import fetch_details
from .ui import draw_boxes, draw_hud, draw_info_panel

SNAPSHOT_DIR = Path("snapshots")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time object detection with Wikipedia details.")
    p.add_argument(
        "--source",
        default="0",
        help=(
            "Video source. Either a webcam index ('0', '1', ...) or a stream URL "
            "such as 'http://PHONE_IP:8080/video' (IP Webcam Android app) or an RTSP URL."
        ),
    )
    p.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights path.")
    p.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    p.add_argument("--width", type=int, default=1280, help="Capture width (local webcams only).")
    p.add_argument("--height", type=int, default=720, help="Capture height (local webcams only).")
    return p.parse_args()


def _resolve_source(src: str):
    return int(src) if src.isdigit() else src


def main() -> int:
    args = parse_args()

    detector = Detector(weights=args.weights, conf=args.conf)

    source = _resolve_source(args.source)
    print(f"[info] opening video source: {source!r}")
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(
            f"ERROR: could not open video source {source!r}.\n"
            "  - For a local webcam, try --source 0 or --source 1.\n"
            "  - For an Android phone, install the free 'IP Webcam' app, tap\n"
            "    'Start server', and use --source http://PHONE_IP:8080/video"
        )
        return 1

    window = "Object Detector (press 'd' for details, 'q' to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    selected_idx = 0
    active_details = None
    frame_times: deque = deque(maxlen=30)

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("WARN: dropped frame")
                continue

            detections = detector.detect(frame)
            if not detections:
                selected_idx = 0
            else:
                selected_idx = max(0, min(selected_idx, len(detections) - 1))

            draw_boxes(frame, detections, selected_idx)
            draw_info_panel(frame, active_details)

            frame_times.append(time.time() - t0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            selected_label = detections[selected_idx].label if detections else None
            draw_hud(frame, fps, selected_label, len(detections))

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d") and detections:
                label = detections[selected_idx].label
                print(f"[info] fetching details for: {label}")
                active_details = fetch_details(label)
            elif key == ord("n") and detections:
                selected_idx = (selected_idx + 1) % len(detections)
            elif key == ord("p") and detections:
                selected_idx = (selected_idx - 1) % len(detections)
            elif key == ord("c"):
                active_details = None
            elif key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                fname = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), frame)
                print(f"[info] saved {fname}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
