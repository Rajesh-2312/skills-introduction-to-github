from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixel coords
    area: int


class Detector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.4):
        self.model = YOLO(weights)
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=self.names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    box=(x1, y1, x2, y2),
                    area=max(0, (x2 - x1)) * max(0, (y2 - y1)),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
