from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixel coords
    area: int
    track_id: Optional[int] = None


class Detector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.4, track: bool = True):
        self.model = YOLO(weights)
        self.conf = conf
        self.track = track
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.track:
            results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)
        else:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        ids = boxes.id
        for i, b in enumerate(boxes):
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            track_id: Optional[int] = None
            if ids is not None:
                try:
                    track_id = int(ids[i].item())
                except (IndexError, RuntimeError):
                    track_id = None
            detections.append(
                Detection(
                    label=self.names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    box=(x1, y1, x2, y2),
                    area=max(0, (x2 - x1)) * max(0, (y2 - y1)),
                    track_id=track_id,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
