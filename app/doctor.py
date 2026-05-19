"""Self-diagnostic for the object-detector app.

Run with: python -m app.doctor

Checks Python version, every optional and required dependency, and external
binaries (espeak, tesseract). Exits 0 when all REQUIRED checks pass.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

REQUIRED = "REQUIRED"
OPTIONAL = "OPTIONAL"

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


@dataclass
class CheckResult:
    name: str
    tier: str  # REQUIRED or OPTIONAL
    status: str  # OK / WARN / FAIL
    detail: str


def _check_python() -> CheckResult:
    v = sys.version_info
    if v >= (3, 9):
        return CheckResult("Python >= 3.9", REQUIRED, OK, f"{v.major}.{v.minor}.{v.micro}")
    return CheckResult("Python >= 3.9", REQUIRED, FAIL, f"found {v.major}.{v.minor}")


def _check_import(module: str, tier: str, hint: str = "") -> CheckResult:
    try:
        importlib.import_module(module)
        return CheckResult(f"import {module}", tier, OK, "")
    except Exception as e:
        return CheckResult(f"import {module}", tier, FAIL if tier == REQUIRED else WARN, f"{e}. {hint}")


def _check_binary(binary: str, tier: str, hint: str = "") -> CheckResult:
    path = shutil.which(binary)
    if path:
        return CheckResult(f"{binary} binary", tier, OK, path)
    return CheckResult(f"{binary} binary", tier, FAIL if tier == REQUIRED else WARN, f"not on PATH. {hint}")


def _check_camera() -> CheckResult:
    try:
        import cv2
    except Exception as e:
        return CheckResult("camera (cv2 open 0)", REQUIRED, FAIL, f"opencv import failed: {e}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return CheckResult(
            "camera (cv2 open 0)", REQUIRED, FAIL,
            "could not open index 0. Try --source 1, or use IP Webcam URL.",
        )
    ok, _ = cap.read()
    cap.release()
    if not ok:
        return CheckResult("camera (cv2 read)", REQUIRED, WARN, "opened but read returned no frame")
    return CheckResult("camera (cv2 read)", REQUIRED, OK, "read 1 frame from index 0")


def _check_mic() -> CheckResult:
    try:
        import sounddevice as sd
    except Exception as e:
        return CheckResult("microphone", OPTIONAL, WARN, f"sounddevice missing: {e}")
    try:
        devices = sd.query_devices()
    except Exception as e:
        return CheckResult("microphone", OPTIONAL, WARN, f"query_devices failed: {e}")
    inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
    if not inputs:
        return CheckResult("microphone", OPTIONAL, WARN, "no input devices found")
    return CheckResult("microphone", OPTIONAL, OK, f"{len(inputs)} input device(s)")


def _check_groq_key() -> CheckResult:
    if os.environ.get("GROQ_API_KEY"):
        return CheckResult("GROQ_API_KEY env var", OPTIONAL, OK, "set")
    return CheckResult(
        "GROQ_API_KEY env var", OPTIONAL, WARN,
        "not set. Free key at https://console.groq.com - needed only for LLM Q&A.",
    )


def _check_yolo_weights() -> CheckResult:
    # YOLOv8 weights download on first use; just check ultralytics is importable.
    try:
        from ultralytics import YOLO  # noqa: F401
        return CheckResult("YOLOv8 weights", REQUIRED, OK, "auto-downloaded on first run if missing")
    except Exception as e:
        return CheckResult("YOLOv8 weights", REQUIRED, FAIL, f"ultralytics import failed: {e}")


def _check_wikipedia() -> CheckResult:
    try:
        import requests
        r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/Cat", timeout=4)
        if r.ok:
            return CheckResult("Wikipedia reachable", OPTIONAL, OK, f"HTTP {r.status_code}")
        return CheckResult("Wikipedia reachable", OPTIONAL, WARN, f"HTTP {r.status_code}")
    except Exception as e:
        return CheckResult("Wikipedia reachable", OPTIONAL, WARN, f"{e} (offline cache will still work)")


CHECKS: List[Callable[[], CheckResult]] = [
    _check_python,
    lambda: _check_import("cv2", REQUIRED, "pip install opencv-python"),
    lambda: _check_import("ultralytics", REQUIRED, "pip install ultralytics"),
    lambda: _check_import("requests", REQUIRED, "pip install requests"),
    lambda: _check_import("pyttsx3", OPTIONAL, "pip install pyttsx3 (or pass --no-tts)"),
    lambda: _check_import("pytesseract", OPTIONAL, "pip install pytesseract (or pass --no-ocr)"),
    lambda: _check_import("transformers", OPTIONAL, "pip install transformers (only for --prompts)"),
    lambda: _check_import("faster_whisper", OPTIONAL, "pip install faster-whisper (or pass --no-voice)"),
    lambda: _check_import("sounddevice", OPTIONAL, "pip install sounddevice (or pass --no-voice)"),
    lambda: _check_import("webrtcvad", OPTIONAL, "pip install webrtcvad (or pass --no-voice)"),
    lambda: _check_import("groq", OPTIONAL, "pip install groq (only needed for LLM Q&A)"),
    lambda: _check_import("flask", OPTIONAL, "pip install flask (only needed for app.web)"),
    lambda: _check_binary("tesseract", OPTIONAL, "apt install tesseract-ocr  /  brew install tesseract"),
    lambda: _check_binary(
        "espeak", OPTIONAL,
        "apt install espeak (Linux only - macOS/Windows have built-in TTS)",
    ),
    _check_yolo_weights,
    _check_camera,
    _check_mic,
    _check_wikipedia,
    _check_groq_key,
]


def main() -> int:
    results: List[CheckResult] = []
    print("Running diagnostics...\n")
    for check in CHECKS:
        try:
            r = check()
        except Exception as e:
            r = CheckResult(check.__name__, REQUIRED, FAIL, f"check raised: {e}")
        results.append(r)
        line = f"{r.status} {r.tier:<8} {r.name}"
        if r.detail:
            line += f"  -  {r.detail}"
        print(line)

    print()
    required_fail = sum(1 for r in results if r.tier == REQUIRED and r.status == FAIL)
    optional_fail = sum(1 for r in results if r.tier == OPTIONAL and r.status in (WARN, FAIL))
    if required_fail:
        print(f"FAILED: {required_fail} required check(s) failed. Fix these before running app.main.")
        return 1
    if optional_fail:
        print(
            f"OK with {optional_fail} optional warning(s). The app will run, but some features "
            "(voice / OCR / TTS / LLM / zero-shot) may be disabled."
        )
    else:
        print("All checks passed. You're ready: python -m app.main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
