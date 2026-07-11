"""Image loading and the deterministic GPDS encoder input transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def load_image(path: Path, max_input_edge: int = 4096) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB").copy()
    if max(image.size) > max_input_edge:
        ratio = max_input_edge / float(max(image.size))
        image = image.resize(
            (round(image.width * ratio), round(image.height * ratio)),
            Image.Resampling.LANCZOS,
        )
    return image


def pil_to_cv(image: Image.Image) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    import cv2

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def otsu_threshold(image: np.ndarray) -> int:
    """Dependency-free Otsu implementation kept identical to training."""
    histogram = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    total = image.size
    sum_total = np.dot(np.arange(256, dtype=np.float64), histogram)
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    threshold = 127
    for level in range(256):
        background_weight += histogram[level]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += level * histogram[level]
        background_mean = background_sum / background_weight
        foreground_mean = (sum_total - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            threshold = level
    return threshold


class SignatureTransform:
    """Match the transform recorded in the GPDS ONNX model metadata.

    Source images use dark ink on a light background. Otsu foreground extraction
    intentionally produces white strokes on a black tensor canvas, matching training.
    """

    def __init__(self) -> None:
        self.mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, source: Path | Image.Image) -> np.ndarray:
        if isinstance(source, Image.Image):
            grayscale = source.convert("L").filter(ImageFilter.GaussianBlur(0.8))
        else:
            with Image.open(source) as image:
                grayscale = image.convert("L").filter(ImageFilter.GaussianBlur(0.8))
        pixels = np.asarray(grayscale, dtype=np.uint8)
        threshold = otsu_threshold(pixels)
        foreground = np.where(pixels <= threshold, 255 - pixels, 0).astype(np.uint8)

        ys, xs = np.nonzero(foreground)
        if len(xs):
            margin = 3
            left = max(0, int(xs.min()) - margin)
            right = min(foreground.shape[1], int(xs.max()) + margin + 1)
            top = max(0, int(ys.min()) - margin)
            bottom = min(foreground.shape[0], int(ys.max()) + margin + 1)
            signature = Image.fromarray(foreground[top:bottom, left:right])
        else:
            signature = Image.fromarray(foreground)

        signature.thumbnail((236, 236), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (256, 256), color=0)
        canvas.paste(signature, ((256 - signature.width) // 2, (256 - signature.height) // 2))
        canvas = canvas.resize((224, 224), Image.Resampling.LANCZOS)

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = np.repeat(array[None, :, :], 3, axis=0)
        return np.ascontiguousarray((tensor - self.mean) / self.std, dtype=np.float32)

    def preview(self, image: Image.Image) -> Image.Image:
        tensor = self(image)
        preview = np.clip(tensor * self.std + self.mean, 0.0, 1.0)
        array = (preview[0] * 255).astype("uint8")
        return Image.fromarray(array).convert("RGB")
