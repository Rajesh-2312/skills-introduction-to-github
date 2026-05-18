from typing import List, Optional

import cv2
import numpy as np

from .detector import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 200, 0)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
DARK = (20, 20, 20)


def draw_boxes(frame: np.ndarray, detections: List[Detection], selected_idx: int) -> None:
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.box
        color = YELLOW if i == selected_idx else GREEN
        thickness = 3 if i == selected_idx else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        id_prefix = f"#{det.track_id} " if det.track_id is not None else ""
        tag = f"{id_prefix}{det.label} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 3, y1 - 5), FONT, 0.6, DARK, 1, cv2.LINE_AA)


def _wrap(text: str, max_chars: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            if len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def draw_info_panel(frame: np.ndarray, details: Optional[dict]) -> None:
    if not details:
        return

    h, w = frame.shape[:2]
    panel_w = int(w * 0.42)
    x0 = w - panel_w - 10
    y0 = 10
    x1 = w - 10
    y1 = h - 50

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), YELLOW, 1)

    title = details.get("title") or details.get("label", "")
    summary = details.get("summary", "")
    url = details.get("url", "")

    pad = 14
    cv2.putText(frame, title, (x0 + pad, y0 + 28), FONT, 0.75, YELLOW, 2, cv2.LINE_AA)

    max_chars = max(20, panel_w // 9)
    text_lines = _wrap(summary, max_chars)
    y = y0 + 58
    line_h = 20
    max_lines = max(1, (y1 - y - 30) // line_h)
    for line in text_lines[:max_lines]:
        cv2.putText(frame, line, (x0 + pad, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        y += line_h

    if len(text_lines) > max_lines:
        cv2.putText(frame, "...", (x0 + pad, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    if url:
        cv2.putText(frame, url[:max_chars], (x0 + pad, y1 - 10), FONT, 0.4, YELLOW, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    selected: Optional[str],
    total: int,
    lang: str = "en",
    tts: str = "off",
) -> None:
    h, w = frame.shape[:2]
    text = (
        f"FPS {fps:4.1f}  |  objects: {total}  |  selected: {selected or '-'}"
        f"  |  lang: {lang}  |  tts: {tts}"
    )
    cv2.putText(frame, text, (10, 24), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)

    controls = "[d] details  [n/p] next/prev  [m] mute  [r] replay  [c] clear  [s] snap  [q] quit"
    cv2.putText(frame, controls, (10, h - 14), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
