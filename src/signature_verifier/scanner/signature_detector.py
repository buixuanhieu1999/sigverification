"""ONNX-only signature detection, crop enhancement, and ink cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .onnx_yolo import get_yolo_detector
from .scan_engine import apply_effect


@dataclass(frozen=True)
class SignatureDetectionResult:
    annotated_image: np.ndarray
    crops: list[np.ndarray]
    detections: list[dict[str, Any]]
    runtime: dict[str, Any]


def isolate_signature_ink(
    image: np.ndarray,
    top_cut_ratio: float = 0.25,
    dilate_kernel: tuple[int, int] = (35, 9),
    min_component_area: int = 15,
    keep_largest_groups: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep likely handwritten strokes on white while discarding small components."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if not 0.0 <= top_cut_ratio <= 1.0:
        raise ValueError("top_cut_ratio must be between 0 and 1")
    if keep_largest_groups < 1:
        raise ValueError("keep_largest_groups must be greater than zero")

    original = image.copy()
    height = image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cut_y = int(height * top_cut_ratio)
    if cut_y > 0:
        ink_mask[:cut_y, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    grouped = cv2.dilate(ink_mask, kernel, iterations=1)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        grouped, connectivity=8
    )

    candidates: list[tuple[float, int]] = []
    for label_id in range(1, label_count):
        x, y, component_width, component_height, area = stats[label_id]
        if area < min_component_area:
            continue
        if y + component_height / 2 < height * top_cut_ratio:
            continue
        score = float(area + component_width * 2 + component_height)
        candidates.append((score, label_id))

    if not candidates:
        return original, ink_mask, grouped

    candidates.sort(reverse=True)
    group_mask = np.zeros_like(ink_mask)
    for _, label_id in candidates[:keep_largest_groups]:
        component_mask = (labels == label_id).astype(np.uint8) * 255
        group_mask = cv2.bitwise_or(group_mask, component_mask)

    final_mask = cv2.bitwise_and(ink_mask, group_mask)
    clean_count, clean_labels, clean_stats, _ = cv2.connectedComponentsWithStats(
        final_mask, connectivity=8
    )
    clean_mask = np.zeros_like(final_mask)
    for label_id in range(1, clean_count):
        if clean_stats[label_id, cv2.CC_STAT_AREA] >= min_component_area:
            clean_mask[clean_labels == label_id] = 255

    result = np.full_like(original, 255)
    result[clean_mask > 0] = original[clean_mask > 0]
    return result, clean_mask, grouped


def _require_onnx_model(model_path: str | Path | None) -> Path:
    if model_path is None:
        raise ValueError("An ONNX signature detector path is required")
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Signature detector not found: {path}")
    if path.suffix.lower() != ".onnx":
        raise ValueError("Signature detection supports ONNX models only")
    return path


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    pad_ratio: float,
    shrink_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    shrink_x = min(box_width * 0.45, box_width * shrink_ratio)
    shrink_y = min(box_height * 0.45, box_height * shrink_ratio)
    x1, y1, x2, y2 = x1 + shrink_x, y1 + shrink_y, x2 - shrink_x, y2 - shrink_y
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width - 1, int(round(x2 + pad_x))),
        min(height - 1, int(round(y2 + pad_y))),
    )


def _draw_detection(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    score: float,
    index: int,
) -> None:
    x1, y1, x2, y2 = box
    color = (28, 90, 230)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    text = f"{index}: {label} {score:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    text_y = max(text_height + 8, y1)
    cv2.rectangle(
        image,
        (x1, text_y - text_height - baseline - 8),
        (min(image.shape[1] - 1, x1 + text_width + 10), text_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 5, text_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def detect_and_crop_signatures(
    image: np.ndarray,
    *,
    model_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 960,
    pad_ratio: float = 0.02,
    shrink_ratio: float = 0.04,
    crop_effect: str = "magic",
    postprocess: bool = True,
    postprocess_top_cut_ratio: float = 0.25,
    postprocess_dilate_kernel: tuple[int, int] = (35, 9),
    postprocess_min_component_area: int = 15,
    postprocess_keep_largest_groups: int = 3,
) -> SignatureDetectionResult:
    """Detect all signature boxes and return enhanced in-memory crops."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    model_file = _require_onnx_model(model_path)
    detector = get_yolo_detector(model_file)
    raw_detections = detector.predict(image, imgsz=imgsz, conf=conf)
    height, width = image.shape[:2]
    annotated = image.copy()
    crops: list[np.ndarray] = []
    detections: list[dict[str, Any]] = []

    rows = []
    for item in raw_detections:
        crop_box = _clip_box(item.box, width, height, pad_ratio, shrink_ratio)
        area = max(0, crop_box[2] - crop_box[0]) * max(0, crop_box[3] - crop_box[1])
        rows.append((crop_box[1], crop_box[0], item, crop_box, area))
    rows.sort(key=lambda row: (row[0], row[1]))

    for index, (_, _, item, crop_box, area) in enumerate(rows, start=1):
        x1, y1, x2, y2 = crop_box
        crop = image[y1 : y2 + 1, x1 : x2 + 1].copy()
        if crop_effect and crop_effect.lower() != "original":
            crop = apply_effect(crop, crop_effect)
        if postprocess:
            crop, _, _ = isolate_signature_ink(
                crop,
                top_cut_ratio=postprocess_top_cut_ratio,
                dilate_kernel=postprocess_dilate_kernel,
                min_component_area=postprocess_min_component_area,
                keep_largest_groups=postprocess_keep_largest_groups,
            )
        crops.append(crop)
        _draw_detection(annotated, crop_box, item.class_name, item.confidence, index)
        detections.append(
            {
                "index": index,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "confidence": round(item.confidence, 4),
                "box": [round(value, 2) for value in item.box],
                "crop_box": list(crop_box),
                "area": int(area),
            }
        )

    return SignatureDetectionResult(
        annotated_image=annotated,
        crops=crops,
        detections=detections,
        runtime={"backend": "onnxruntime", **detector.info()},
    )
