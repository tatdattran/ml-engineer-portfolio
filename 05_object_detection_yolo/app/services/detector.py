from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float, iou_threshold: float) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model: YOLO | None = None

    def load(self) -> None:
        if self.model is None:
            logger.info("Loading YOLO model from %s", self.model_path)
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        results = self.model.predict(
            source=image_np,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        result = results[0]

        detections: list[Detection] = []
        names = result.names

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            bbox_xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names[class_id]),
                    confidence=round(confidence, 4),
                    bbox_xyxy=[round(v, 2) for v in bbox_xyxy],
                )
            )

        return {
            "image_width": int(image.width),
            "image_height": int(image.height),
            "model": self.model_path,
            "detections": [d.__dict__ for d in detections],
            "count": len(detections),
        }
