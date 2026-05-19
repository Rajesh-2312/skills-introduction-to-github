from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from .detector import Detection


class ZeroShotDetector:
    """OWL-ViT v2 open-vocabulary detector. Detects whatever the user names."""

    def __init__(
        self,
        prompts: List[str],
        model_id: str = "google/owlv2-base-patch16-ensemble",
        conf: float = 0.15,
        device: Optional[str] = None,
    ):
        prompts = [p.strip() for p in prompts if p and p.strip()]
        if not prompts:
            raise ValueError("ZeroShotDetector needs at least one prompt.")
        self.prompts = prompts
        self.conf = conf
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[zero-shot] loading {model_id} on {self.device} (first run downloads ~500 MB)")
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        rgb = frame[..., ::-1]  # BGR -> RGB
        image = Image.fromarray(rgb)
        h, w = frame.shape[:2]

        with torch.no_grad():
            inputs = self.processor(
                text=[self.prompts], images=image, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(h, w)], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.conf
        )[0]

        detections: List[Detection] = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            detections.append(
                Detection(
                    label=self.prompts[int(label_idx.item())],
                    confidence=float(score.item()),
                    box=(x1, y1, x2, y2),
                    area=max(0, x2 - x1) * max(0, y2 - y1),
                    track_id=None,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
