import argparse
import queue
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2

from . import ocr
from .detector import Detector
from .info_fetcher import fetch_details
from .llm import GroqClient
from .speech import Speaker, summarize_for_speech
from .ui import draw_boxes, draw_hud, draw_info_panel
from .voice import VoiceListener

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
    p.add_argument("--tts-voice", default=None, help="pyttsx3 voice id (OS-specific).")
    p.add_argument("--tts-rate", type=int, default=175, help="Speech rate in words per minute.")
    p.add_argument(
        "--prompts",
        default=None,
        help=(
            "Comma-separated zero-shot prompts. When set, swaps YOLOv8 for OWL-ViT and "
            "detects anything matching these prompts. Example: "
            "--prompts 'wine bottle, guitar, red shoe'. Slow on CPU."
        ),
    )
    p.add_argument("--zs-conf", type=float, default=0.15, help="Zero-shot confidence threshold.")
    p.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable Tesseract OCR fallback on the selected object.",
    )
    p.add_argument("--no-voice", action="store_true", help="Disable voice command listener.")
    p.add_argument(
        "--voice-model",
        default="base",
        help="faster-whisper model size: tiny, base, small, medium (default: base).",
    )
    p.add_argument("--voice-lang", default="en", help="Voice transcription language code.")
    p.add_argument(
        "--llm-model",
        default="llama-3.1-8b-instant",
        help="Groq model for follow-up Q&A.",
    )
    return p.parse_args()


def _resolve_source(src: str):
    return int(src) if src.isdigit() else src


def main() -> int:
    args = parse_args()

    if args.prompts:
        from .zero_shot import ZeroShotDetector

        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        detector = ZeroShotDetector(prompts=prompts, conf=args.zs_conf)
        model_name = "OWL-ViT"
    else:
        detector = Detector(weights=args.weights, conf=args.conf, track=not args.no_track)
        model_name = "YOLOv8"

    speaker = Speaker(enabled=not args.no_tts, voice_id=args.tts_voice, rate=args.tts_rate)
    ocr_enabled = not args.no_ocr and ocr.is_available()
    if not args.no_ocr and not ocr_enabled:
        print("[ocr] tesseract not found - install it or pass --no-ocr to silence this message.")

    llm = GroqClient(model=args.llm_model)
    command_q: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()

    voice = None
    if not args.no_voice:
        voice = VoiceListener(
            callback=lambda cmd, arg: command_q.put((cmd, arg)),
            model_size=args.voice_model,
            language=args.voice_lang,
        )

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

    state = {
        "selected_idx": 0,
        "active_details": None,
        "last_fetched_key": None,
        "should_quit": False,
    }
    frame_times: deque = deque(maxlen=30)

    def cmd_details(_arg: Optional[str], frame, detections) -> None:
        if not detections:
            return
        sel = detections[state["selected_idx"]]
        key_tuple = (sel.track_id, sel.label)
        if key_tuple == state["last_fetched_key"] and state["active_details"]:
            return
        print(f"[info] fetching details for: {sel.label} (lang={args.lang})")
        details = fetch_details(sel.label, lang=args.lang)
        if ocr_enabled:
            x1, y1, x2, y2 = sel.box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None
            text = ocr.extract_text(crop) if crop is not None else ""
            if text:
                print(f"[ocr] {text!r}")
            details["ocr_text"] = text
        state["active_details"] = details
        state["last_fetched_key"] = key_tuple
        spoken = summarize_for_speech(details.get("title", ""), details.get("summary", ""))
        if details.get("ocr_text"):
            spoken += f" Text on object: {details['ocr_text']}."
        speaker.speak(spoken)

    def cmd_ask(arg: Optional[str], *_unused) -> None:
        if not arg or not arg.strip():
            speaker.speak("What would you like to ask?")
            return
        if not state["active_details"]:
            print("[llm] no object details active. Say 'what is this' first.")
            speaker.speak("Please ask for details first.")
            return
        if not llm.available:
            speaker.speak("Question answering is not available.")
            return
        print(f"[llm] question: {arg!r}")
        answer = llm.ask(arg, state["active_details"])
        print(f"[llm] answer: {answer}")
        state["active_details"] = dict(state["active_details"])
        state["active_details"]["summary"] = f"Q: {arg}\nA: {answer}\n\n" + state["active_details"].get("summary", "")
        speaker.speak(answer)

    def dispatch(cmd: str, arg: Optional[str], frame, detections) -> None:
        if cmd == "quit":
            state["should_quit"] = True
        elif cmd == "details":
            cmd_details(arg, frame, detections)
        elif cmd == "ask":
            cmd_ask(arg, frame, detections)
        elif cmd == "replay" and state["active_details"]:
            d = state["active_details"]
            speaker.speak(summarize_for_speech(d.get("title", ""), d.get("summary", "")))
        elif cmd == "mute":
            muted = speaker.toggle_mute()
            print(f"[tts] {'muted' if muted else 'unmuted'}")
        elif cmd == "next" and detections:
            state["selected_idx"] = (state["selected_idx"] + 1) % len(detections)
        elif cmd == "prev" and detections:
            state["selected_idx"] = (state["selected_idx"] - 1) % len(detections)
        elif cmd == "clear":
            state["active_details"] = None
            state["last_fetched_key"] = None
        elif cmd == "snapshot":
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            fname = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
            cv2.imwrite(str(fname), frame)
            print(f"[info] saved {fname}")
        elif cmd == "unknown":
            # Free-form sentence from voice. Treat as LLM question when details are showing.
            if state["active_details"] and llm.available and arg:
                cmd_ask(arg, frame, detections)

    key_to_cmd = {
        ord("q"): "quit",
        ord("d"): "details",
        ord("r"): "replay",
        ord("m"): "mute",
        ord("n"): "next",
        ord("p"): "prev",
        ord("c"): "clear",
        ord("s"): "snapshot",
    }

    try:
        while not state["should_quit"]:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("WARN: dropped frame")
                continue

            detections = detector.detect(frame)
            if not detections:
                state["selected_idx"] = 0
            else:
                state["selected_idx"] = max(0, min(state["selected_idx"], len(detections) - 1))

            draw_boxes(frame, detections, state["selected_idx"])
            draw_info_panel(frame, state["active_details"])

            frame_times.append(time.time() - t0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            if detections:
                sel = detections[state["selected_idx"]]
                id_prefix = f"#{sel.track_id} " if sel.track_id is not None else ""
                selected_label = f"{id_prefix}{sel.label}"
            else:
                selected_label = None
            tts_state = "off" if not speaker.available else ("muted" if speaker.muted else "on")
            draw_hud(
                frame, fps, selected_label, len(detections),
                lang=args.lang, tts=tts_state, model=model_name,
            )

            cv2.imshow(window, frame)

            # Drain voice command queue first.
            while True:
                try:
                    vcmd, varg = command_q.get_nowait()
                except queue.Empty:
                    break
                dispatch(vcmd, varg, frame, detections)

            key = cv2.waitKey(1) & 0xFF
            mapped = key_to_cmd.get(key)
            if mapped:
                dispatch(mapped, None, frame, detections)
    finally:
        if voice:
            voice.shutdown()
        speaker.shutdown()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
