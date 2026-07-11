"""Signature extraction independent of any web framework."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .config import AppSettings
from .domain import InputMode, SignatureCandidate, SourceBundle
from .preprocessing import cv_to_pil, load_image, pil_to_cv
from .scanner.scan_engine import apply_effect, scan_document
from .scanner.signature_detector import detect_and_crop_signatures, isolate_signature_ink
from .scanner.onnx_yolo import get_yolo_detector


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_paths(files: Sequence[str | Path] | str | Path | None) -> list[Path]:
    if files is None:
        return []
    values: Iterable[object] = [files] if isinstance(files, (str, Path)) else files
    paths: list[Path] = []
    for value in values:
        if value is None:
            continue
        path = value if isinstance(value, Path) else Path(getattr(value, "name", value))
        path = path.expanduser().resolve()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return paths


class SignatureExtractor:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        # Load both detectors during application startup so /health/ready is truthful.
        self.document_detector = get_yolo_detector(settings.document_model_path)
        self.signature_detector = get_yolo_detector(settings.signature_model_path)

    def extract_source(
        self,
        path: Path,
        *,
        role: str,
        source_number: int,
        mode: InputMode,
        signature_confidence: float,
        signature_padding: float,
        signature_shrink: float,
        query_top_cut: float,
    ) -> SourceBundle:
        source = load_image(path, self.settings.max_input_edge)
        bundle = SourceBundle(role=role, source_name=path.name, source_number=source_number)

        if mode is InputMode.CROPPED_SIGNATURE:
            # A cropped signature has no printed document heading to remove. Apply
            # the same zero top-cut on both sides so verification stays symmetric.
            top_cut = 0.0
            scanned_crop = apply_effect(pil_to_cv(source), "magic")
            cleaned_crop, _, _ = isolate_signature_ink(
                scanned_crop,
                top_cut_ratio=top_cut,
                dilate_kernel=(35, 9),
                min_component_area=15,
                keep_largest_groups=self.settings.keep_largest_ink_groups,
            )
            bundle.candidates.append(
                SignatureCandidate(
                    role=role,
                    source_name=path.name,
                    source_number=source_number,
                    signature_number=1,
                    image=cv_to_pil(cleaned_crop),
                    input_mode=mode,
                )
            )
            bundle.metadata = {
                "input_mode": mode.value,
                "document_detection": "skipped: source is an already-cropped signature",
                "crop_effect": "magic",
                "postprocess": "isolate_signature_ink",
                "postprocess_top_cut": top_cut,
                "postprocess_keep_largest_groups": self.settings.keep_largest_ink_groups,
                "signature_count": 1,
            }
            return bundle

        self.settings.validate(require_detectors=True)
        scan_result = scan_document(
            pil_to_cv(source),
            effect="simple",
            detector="yolo",
            yolo_model=self.settings.document_model_path,
            yolo_conf=0.05,
            yolo_imgsz=960,
            yolo_classes="book",
            yolo_expand=0.08,
            yolo_refine=True,
        )
        signature_result = detect_and_crop_signatures(
            scan_result.warped_image,
            model_path=self.settings.signature_model_path,
            conf=float(signature_confidence),
            imgsz=960,
            pad_ratio=float(signature_padding),
            shrink_ratio=float(signature_shrink),
            crop_effect="magic",
            postprocess=True,
            postprocess_top_cut_ratio=float(query_top_cut),
            postprocess_dilate_kernel=(35, 9),
            postprocess_min_component_area=15,
            postprocess_keep_largest_groups=self.settings.keep_largest_ink_groups,
        )
        bundle.warped_document = cv_to_pil(scan_result.warped_image)
        bundle.detection_overlay = cv_to_pil(signature_result.annotated_image)
        warnings = []
        if not scan_result.document_found:
            warnings.append("Document border not confidently found; scanner fallback warp used.")
        bundle.metadata = {
            "input_mode": mode.value,
            "document_found": bool(scan_result.document_found),
            "document_detection": scan_result.detection,
            "signature_count": len(signature_result.crops),
            "signature_detections": signature_result.detections,
            "signature_detector_runtime": signature_result.runtime,
            "signature_source": "original warped document",
            "crop_effect": "magic",
            "postprocess": "isolate_signature_ink",
            "signature_confidence": float(signature_confidence),
            "signature_padding": float(signature_padding),
            "signature_shrink": float(signature_shrink),
            "query_top_cut": float(query_top_cut),
            "postprocess_keep_largest_groups": self.settings.keep_largest_ink_groups,
            "warnings": warnings,
        }
        for number, crop in enumerate(signature_result.crops, start=1):
            if crop is None or crop.size == 0:
                continue
            detection = (
                signature_result.detections[number - 1]
                if number <= len(signature_result.detections)
                else None
            )
            bundle.candidates.append(
                SignatureCandidate(
                    role=role,
                    source_name=path.name,
                    source_number=source_number,
                    signature_number=number,
                    image=cv_to_pil(crop),
                    detection=detection,
                    input_mode=mode,
                )
            )
        return bundle

    def extract_many(
        self,
        files: Sequence[str | Path] | str | Path | None,
        *,
        role: str,
        mode: InputMode,
        signature_confidence: float,
        signature_padding: float,
        signature_shrink: float,
        query_top_cut: float,
    ) -> tuple[list[SourceBundle], list[dict[str, str]]]:
        bundles: list[SourceBundle] = []
        errors: list[dict[str, str]] = []
        for number, path in enumerate(normalize_paths(files), start=1):
            try:
                bundles.append(
                    self.extract_source(
                        path,
                        role=role,
                        source_number=number,
                        mode=mode,
                        signature_confidence=signature_confidence,
                        signature_padding=signature_padding,
                        signature_shrink=signature_shrink,
                        query_top_cut=query_top_cut,
                    )
                )
            except Exception as exc:
                errors.append(
                    {"source": path.name, "error": type(exc).__name__, "message": str(exc)}
                )
        return bundles, errors
