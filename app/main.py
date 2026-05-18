import argparse
import time
from collections import deque
from pathlib import Path

import cv2

from .detector import Detector
from .info_fetcher import fetch_details
from .speech import Speaker, summarize_for_speech
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
    p.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language code for details (en, hi, te, ta, fr, es, ...). Default: en.",
    )
    p.add_argument("--no-track", action="store_true", help="Disable object tracking IDs.")
    p.add_argument("--no-tts", action="store_true", help="Disable text-to-speech readout.")
    p.add_argument("--voice", default=None, help="pyttsx3 voice id (OS-specific). Default: system voice.")
    p.add_argument("--tts-rate", type=int, default=175, help="Speech rate in words per minute.")
    return p.parse_args()


def _resolve_source(src: str):
    return int(src) if src.isdigit() else src


def main() -> int:
    args = parse_args()

    detector = Detector(weights=args.weights, conf=args.conf, track=not args.no_track)
    speaker = Speaker(enabled=not args.no_tts, voice_id=args.voice, rate=args.tts_rate)

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
    last_fetched_key = None  # (track_id, label) - skip refetch if user holds 'd'
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
            if detections:
                sel = detections[selected_idx]
                id_prefix = f"#{sel.track_id} " if sel.track_id is not None else ""
                selected_label = f"{id_prefix}{sel.label}"
            else:
                selected_label = None
            tts_state = "off" if not speaker.available else ("muted" if speaker.muted else "on")
            draw_hud(frame, fps, selected_label, len(detections), lang=args.lang, tts=tts_state)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d") and detections:
                sel = detections[selected_idx]
                key_tuple = (sel.track_id, sel.label)
                if key_tuple == last_fetched_key and active_details:
                    pass  # already showing this object's details
                else:
                    print(f"[info] fetching details for: {sel.label} (lang={args.lang})")
                    active_details = fetch_details(sel.label, lang=args.lang)
                    last_fetched_key = key_tuple
                    speaker.speak(
                        summarize_for_speech(
                            active_details.get("title", ""),
                            active_details.get("summary", ""),
                        )
                    )
            elif key == ord("r") and active_details:
                speaker.speak(
                    summarize_for_speech(
                        active_details.get("title", ""),
                        active_details.get("summary", ""),
                    )
                )
            elif key == ord("m"):
                muted = speaker.toggle_mute()
                print(f"[tts] {'muted' if muted else 'unmuted'}")
            elif key == ord("n") and detections:
                selected_idx = (selected_idx + 1) % len(detections)
            elif key == ord("p") and detections:
                selected_idx = (selected_idx - 1) % len(detections)
            elif key == ord("c"):
                active_details = None
                last_fetched_key = None
            elif key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                fname = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), frame)
                print(f"[info] saved {fname}")
    finally:
        speaker.shutdown()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
