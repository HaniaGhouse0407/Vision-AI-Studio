"""
YOLOv8 Object Detection Pipeline
Wraps Ultralytics YOLOv8 with NMS, tracking, and export utilities.
"""
from __future__ import annotations
import cv2, numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    track_id: Optional[int] = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class YOLODetector:
    """
    Production YOLOv8 detector with confidence filtering, NMS, and annotation.
    Supports image, video, and webcam streams.
    """

    COCO_PALETTE = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ]

    def __init__(
        self,
        model_size: str = "n",       # n, s, m, l, x
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.filter_classes = classes
        self.model = self._load_model(model_size)

    def _load_model(self, size: str):
        try:
            from ultralytics import YOLO
            model_path = f"yolov8{size}.pt"
            logger.info(f"Loading {model_path}...")
            return YOLO(model_path)
        except ImportError:
            raise ImportError("pip install ultralytics")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame."""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.filter_classes,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append(Detection(
                class_id=cls_id,
                class_name=self.model.names[cls_id],
                confidence=float(box.conf[0]),
                bbox=tuple(map(int, box.xyxy[0].tolist())),
            ))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch inference for efficiency."""
        results = self.model(
            frames, conf=self.conf_threshold, iou=self.iou_threshold,
            device=self.device, verbose=False
        )
        return [
            [Detection(
                class_id=int(b.cls[0]),
                class_name=self.model.names[int(b.cls[0])],
                confidence=float(b.conf[0]),
                bbox=tuple(map(int, b.xyxy[0].tolist())),
            ) for b in r.boxes]
            for r in results
        ]

    def annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        out = frame.copy()
        for det in detections:
            color = self.COCO_PALETTE[det.class_id % len(self.COCO_PALETTE)]
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"#{det.track_id} " + label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return out

    def count_by_class(self, detections: List[Detection]) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(d.class_name for d in detections))
