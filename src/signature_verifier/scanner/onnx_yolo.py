"""Small YOLOv8 ONNX detector using only ONNX Runtime, NumPy, and OpenCV."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..ort_session import create_session, run_with_cpu_fallback


@dataclass(frozen=True)
class YoloDetection:
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def _parse_names(metadata: dict[str, str]) -> dict[int, str]:
    raw = metadata.get("names", "{}")
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = {}
    return {int(key): str(value) for key, value in parsed.items()}


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    pad_width = (size - resized_width) / 2
    pad_height = (size - resized_height) / 2
    # Ultralytics rect inference uses the minimum stride-aligned rectangle.
    pad_width = float(np.mod(pad_width * 2, 32)) / 2
    pad_height = float(np.mod(pad_height * 2, 32)) / 2
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_height - 0.1)), int(round(pad_height + 0.1))
    left, right = int(round(pad_width - 0.1)), int(round(pad_width + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor[None], ratio, pad_width, pad_height


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(box_area + areas - intersection, 1e-7)


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        overlaps = _iou(boxes[current], boxes[remaining])
        suppress = (overlaps > iou_threshold) & (classes[remaining] == classes[current])
        order = remaining[~suppress]
    return keep


class YoloOnnxDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve()
        self.session, self.device = create_session(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.names = _parse_names(self.session.get_modelmeta().custom_metadata_map)

    def predict(
        self,
        image: np.ndarray,
        *,
        imgsz: int = 960,
        conf: float = 0.25,
        iou: float = 0.7,
        max_detections: int = 300,
    ) -> list[YoloDetection]:
        tensor, ratio, pad_width, pad_height = _letterbox(image, imgsz)
        self.session, outputs = run_with_cpu_fallback(
            self.session,
            self.model_path,
            [self.output_name],
            {self.input_name: tensor},
        )
        prediction = outputs[0]
        providers = self.session.get_providers()
        self.device = "cuda" if providers and providers[0] == "CUDAExecutionProvider" else "cpu"
        rows = prediction[0]
        if rows.shape[0] < rows.shape[1]:
            rows = rows.T
        class_scores = rows[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(rows)), class_ids]
        selected = scores >= float(conf)
        if not np.any(selected):
            return []
        boxes = _xywh_to_xyxy(rows[selected, :4])
        scores = scores[selected]
        class_ids = class_ids[selected]
        keep = _nms(boxes, scores, class_ids, iou, max_detections)

        height, width = image.shape[:2]
        detections: list[YoloDetection] = []
        for index in keep:
            box = boxes[index].copy()
            box[[0, 2]] = (box[[0, 2]] - pad_width) / ratio
            box[[1, 3]] = (box[[1, 3]] - pad_height) / ratio
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            class_id = int(class_ids[index])
            detections.append(
                YoloDetection(
                    box=tuple(float(value) for value in box),
                    confidence=float(scores[index]),
                    class_id=class_id,
                    class_name=self.names.get(class_id, str(class_id)),
                )
            )
        return detections

    def info(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path),
            "device": self.device,
            "providers": self.session.get_providers(),
            "names": {str(key): value for key, value in self.names.items()},
        }


@lru_cache(maxsize=4)
def get_yolo_detector(model_path: str | Path) -> YoloOnnxDetector:
    return YoloOnnxDetector(Path(model_path))
