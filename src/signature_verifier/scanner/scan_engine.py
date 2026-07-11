"""ONNX-assisted document detection, perspective correction, and enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .onnx_yolo import get_yolo_detector


VALID_EFFECTS = {"magic", "simple", "bw", "original"}
VALID_DETECTORS = {"classic", "yolo"}


@dataclass(frozen=True)
class ScanResult:
    image: np.ndarray
    warped_image: np.ndarray
    corners: np.ndarray
    document_found: bool
    effect: str
    detector: str
    detection: dict[str, Any]


def order_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points_array = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points_array.sum(axis=1)
    differences = np.diff(points_array, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points_array[np.argmin(sums)]
    ordered[2] = points_array[np.argmax(sums)]
    ordered[1] = points_array[np.argmin(differences)]
    ordered[3] = points_array[np.argmax(differences)]
    return ordered


def _resize_for_detection(
    image: np.ndarray, max_edge: int = 1100
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image.copy(), 1.0
    scale = max_edge / float(longest)
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _candidate_quads(edge_image: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = edge_image.shape[0] * edge_image.shape[1]
    quads: list[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:24]:
        if cv2.contourArea(contour) < image_area * 0.05:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            quads.append(approximation.reshape(4, 2).astype(np.float32))
    return quads


def _score_quad(quad: np.ndarray, width: int, height: int) -> float:
    ordered = order_points(quad)
    area_ratio = cv2.contourArea(ordered) / float(width * height)
    sides = [
        np.linalg.norm(ordered[0] - ordered[1]),
        np.linalg.norm(ordered[1] - ordered[2]),
        np.linalg.norm(ordered[2] - ordered[3]),
        np.linalg.norm(ordered[3] - ordered[0]),
    ]
    shape_penalty = max(sides) / max(min(sides), 1.0)
    center_penalty = np.linalg.norm(
        ordered.mean(axis=0) - np.array([width / 2, height / 2])
    ) / max(width, height)
    return float(area_ratio - 0.05 * shape_penalty - 0.2 * center_penalty)


def find_document_corners_classic(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Find a document contour or return a conservative inset fallback."""
    small, scale = _resize_for_detection(image)
    height, width = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.Canny(gray, 45, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    quads = _candidate_quads(edges)
    if not quads:
        threshold = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )
        threshold = cv2.morphologyEx(
            cv2.bitwise_not(threshold), cv2.MORPH_CLOSE, kernel, iterations=2
        )
        quads = _candidate_quads(threshold)
    if quads:
        best = max(quads, key=lambda item: _score_quad(item, width, height))
        return order_points(best / scale), True

    source_height, source_width = image.shape[:2]
    inset_x, inset_y = source_width * 0.04, source_height * 0.04
    return np.array(
        [
            [inset_x, inset_y],
            [source_width - inset_x, inset_y],
            [source_width - inset_x, source_height - inset_y],
            [inset_x, source_height - inset_y],
        ],
        dtype=np.float32,
    ), False


def _expand_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * ratio, (y2 - y1) * ratio
    x1, y1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
    x2, y2 = min(width - 1.0, x2 + pad_x), min(height - 1.0, y2 + pad_y)
    snap_x, snap_y = width * 0.05, height * 0.05
    return (
        0.0 if x1 <= snap_x else x1,
        0.0 if y1 <= snap_y else y1,
        width - 1.0 if width - x2 <= snap_x else x2,
        height - 1.0 if height - y2 <= snap_y else y2,
    )


def _box_corners(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _largest_mask_quad(mask: np.ndarray, offset: tuple[int, int]) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < mask.shape[0] * mask.shape[1] * 0.15:
        return None
    perimeter = cv2.arcLength(contour, True)
    quad: np.ndarray | None = None
    for epsilon in (0.01, 0.015, 0.02, 0.03, 0.05):
        approximation = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            quad = approximation.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    quad[:, 0] += offset[0]
    quad[:, 1] += offset[1]
    return order_points(quad)


def _refine_document_box(
    image: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, {"reason": "empty_roi"}
    crop = image[y1 : y2 + 1, x1 : x2 + 1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([179, 70, 255]))
    long_edge = max(crop.shape[:2])
    close_size = max(21, int(long_edge * 0.018))
    open_size = max(7, int(long_edge * 0.006))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size)),
    )
    quad = _largest_mask_quad(mask, (x1, y1))
    return quad, {"reason": "mask_quad" if quad is not None else "no_mask_quad"}


def find_document_corners_yolo(
    image: np.ndarray,
    model_path: str | Path,
    conf: float = 0.05,
    imgsz: int = 960,
    class_names: str | Iterable[str] | None = None,
    expand: float = 0.08,
    refine: bool = True,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """Use an ONNX detector for the page region, then refine its four corners."""
    model_file = Path(model_path).expanduser().resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Document detector not found: {model_file}")
    if model_file.suffix.lower() != ".onnx":
        raise ValueError("Document detection supports ONNX models only")

    detector = get_yolo_detector(model_file)
    allowed = (
        {"book"}
        if class_names is None
        else {
            item.strip().lower()
            for item in (
                class_names.split(",") if isinstance(class_names, str) else class_names
            )
            if item.strip()
        }
    )
    detections = detector.predict(image, imgsz=imgsz, conf=conf)
    height, width = image.shape[:2]
    records = []
    for item in detections:
        x1, y1, x2, y2 = item.box
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(width * height)
        records.append(
            {
                "class_id": item.class_id,
                "class_name": item.class_name.lower(),
                "confidence": round(item.confidence, 4),
                "box": [round(value, 2) for value in item.box],
                "area_ratio": round(area_ratio, 4),
            }
        )
    targeted = [
        item
        for item in records
        if item["class_name"] in allowed and item["area_ratio"] >= 0.02
    ]
    selected = targeted or sorted(records, key=lambda item: item["area_ratio"], reverse=True)[:1]
    runtime = {"backend": "onnxruntime", **detector.info()}
    if not selected:
        corners, found = find_document_corners_classic(image)
        return corners, found, {
            "detector": "yolo",
            "runtime": runtime,
            "detections": records,
            "fallback": "classic",
        }

    box = _expand_box(
        (
            min(item["box"][0] for item in selected),
            min(item["box"][1] for item in selected),
            max(item["box"][2] for item in selected),
            max(item["box"][3] for item in selected),
        ),
        width,
        height,
        expand,
    )
    info: dict[str, Any] = {
        "detector": "yolo",
        "runtime": runtime,
        "allowed_classes": sorted(allowed),
        "selected_box": [round(value, 2) for value in box],
        "detections": records,
    }
    if refine:
        refined, refine_info = _refine_document_box(image, box)
        info["refine"] = refine_info
        if refined is not None:
            info["source"] = "yolo_mask_refine"
            return refined, True, info
    info["source"] = "yolo_box"
    return _box_corners(box), True, info


def four_point_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = order_points(corners)
    max_width = max(
        2,
        int(
            round(
                max(
                    np.linalg.norm(top_right - top_left),
                    np.linalg.norm(bottom_right - bottom_left),
                )
            )
        ),
    )
    max_height = max(
        2,
        int(
            round(
                max(
                    np.linalg.norm(bottom_right - top_right),
                    np.linalg.norm(bottom_left - top_left),
                )
            )
        ),
    )
    target = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left]), target
    )
    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _auto_contrast(image: np.ndarray) -> np.ndarray:
    adjusted = []
    for channel in cv2.split(image):
        low, high = np.percentile(channel, (1.0, 99.0))
        if high <= low:
            adjusted.append(channel)
            continue
        clipped = np.clip(channel, low, high)
        adjusted.append(((clipped - low) * (255.0 / (high - low))).astype(np.uint8))
    return cv2.merge(adjusted)


def _unsharp(image: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def _magic_effect(image: np.ndarray) -> np.ndarray:
    gray = cv2.fastNlMeansDenoising(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, 8, 7, 21)
    kernel_size = min(max(31, (min(gray.shape[:2]) // 24) | 1), 99)
    background = cv2.medianBlur(gray, kernel_size)
    flattened = cv2.normalize(
        cv2.divide(gray, background, scale=255), None, 0, 255, cv2.NORM_MINMAX
    )
    block = max(31, (min(gray.shape[:2]) // 32) | 1)
    scanned = cv2.adaptiveThreshold(
        flattened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 9
    )
    blended = cv2.addWeighted(flattened, 0.28, scanned, 0.72, 0)
    return cv2.cvtColor(_unsharp(blended, 0.25, 1.0), cv2.COLOR_GRAY2BGR)


def apply_effect(image: np.ndarray, effect: str) -> np.ndarray:
    effect_name = effect.lower()
    if effect_name not in VALID_EFFECTS:
        raise ValueError(f"Unknown scan effect: {effect}")
    if effect_name == "magic":
        return _magic_effect(image)
    if effect_name == "simple":
        denoised = cv2.bilateralFilter(_auto_contrast(image), 7, 40, 40)
        return _unsharp(denoised, 0.35, 1.2)
    if effect_name == "bw":
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        block = max(31, (min(gray.shape[:2]) // 32) | 1)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 11
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return image.copy()


def scan_document(
    image: np.ndarray,
    *,
    effect: str = "magic",
    corners: np.ndarray | None = None,
    detector: str = "yolo",
    yolo_model: str | Path | None = None,
    yolo_conf: float = 0.05,
    yolo_imgsz: int = 960,
    yolo_classes: str | Iterable[str] | None = None,
    yolo_expand: float = 0.08,
    yolo_refine: bool = True,
) -> ScanResult:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if detector not in VALID_DETECTORS:
        raise ValueError(f"Unknown document detector: {detector}")

    detected = True
    detection: dict[str, Any] = {"detector": "manual", "source": "provided_corners"}
    if corners is None and detector == "classic":
        corners, detected = find_document_corners_classic(image)
        detection = {
            "detector": "classic",
            "source": "contour" if detected else "fallback",
        }
    elif corners is None:
        if yolo_model is None:
            raise ValueError("yolo_model is required for ONNX document detection")
        corners, detected, detection = find_document_corners_yolo(
            image,
            yolo_model,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
            class_names=yolo_classes,
            expand=yolo_expand,
            refine=yolo_refine,
        )
    else:
        corners = order_points(corners)

    warped = four_point_transform(image, corners)
    processed = apply_effect(warped, effect)
    return ScanResult(processed, warped, corners, detected, effect, detector, detection)
