"""Optional Gradio demo for quick local GPDS signature verification tests."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .config import AppSettings
from .domain import InputMode, SignatureCandidate, SourceBundle, VerificationReport
from .preprocessing import SignatureTransform
from .service import SignatureVerificationService


COMPARISON_HEADERS = [
    "Query signature",
    "Closest reference",
    "Distance",
    "Similarity %",
    "Required %",
    "Prediction",
    "Hidden refs",
]


def _load_gradio():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised by users without demo extra.
        raise RuntimeError(
            "Gradio is optional. Install/run the demo with: "
            "uv run --extra onnx-cpu --extra demo signature-verifier-demo"
        ) from exc
    return gr


@lru_cache(maxsize=1)
def _service() -> SignatureVerificationService:
    settings = AppSettings.from_env()
    settings.validate(require_detectors=True)
    return SignatureVerificationService(settings)


def _coerce_paths(files: Any) -> list[Path]:
    """Normalize Gradio file values across versions into filesystem paths."""
    if files is None:
        return []
    if isinstance(files, str | Path):
        return [Path(files)]
    if isinstance(files, dict):
        value = files.get("path") or files.get("name")
        return [] if value is None else [Path(value)]
    if isinstance(files, list | tuple):
        paths: list[Path] = []
        for item in files:
            paths.extend(_coerce_paths(item))
        return paths
    name = getattr(files, "name", None) or getattr(files, "path", None)
    return [] if name is None else [Path(name)]


def _candidate_gallery(candidates: list[SignatureCandidate]) -> list[tuple[Image.Image, str]]:
    return [(candidate.image, candidate.label) for candidate in candidates]


def _model_input_gallery(candidates: list[SignatureCandidate]) -> list[tuple[Image.Image, str]]:
    transform = SignatureTransform()
    return [(transform.preview(candidate.image), candidate.label) for candidate in candidates]


def _document_gallery(bundles: list[SourceBundle]) -> list[tuple[Image.Image, str]]:
    images: list[tuple[Image.Image, str]] = []
    for bundle in bundles:
        if bundle.warped_document is not None:
            images.append((bundle.warped_document, f"{bundle.source_name} · scanned document"))
        if bundle.detection_overlay is not None:
            images.append((bundle.detection_overlay, f"{bundle.source_name} · signature detections"))
    return images


def _comparison_rows(report: VerificationReport) -> list[list[Any]]:
    return [
        [
            row.query,
            row.closest_reference,
            round(row.distance, 6),
            round(row.similarity_percent, 2),
            round(row.signature_rate, 2),
            "match" if row.matched else "no_match",
            row.hidden_reference_count,
        ]
        for row in report.rows
    ]


def _status_markdown(report: VerificationReport, elapsed_ms: float) -> str:
    nearest = (
        "n/a"
        if report.nearest_similarity_percent is None
        else f"{report.nearest_similarity_percent:.2f}%"
    )
    distance = "n/a" if report.nearest_distance is None else f"{report.nearest_distance:.6f}"
    icon = {
        "all_match": "✅",
        "mixed": "⚠️",
        "no_match": "❌",
        "failed": "🛑",
    }.get(report.status, "ℹ️")
    return (
        f"### {icon} {report.status}\n\n"
        f"{report.message}\n\n"
        f"- Nearest similarity: `{nearest}`\n"
        f"- Nearest distance: `{distance}`\n"
        f"- Required signature rate: `{report.signature_rate:.2f}%`\n"
        f"- Processing time: `{elapsed_ms:.0f} ms`"
    )


def verify_with_gradio(
    references: Any,
    queries: Any,
    query_mode: str,
    signature_rate: float,
    signature_confidence: float,
    signature_padding: float,
    signature_shrink: float,
    query_top_cut: float,
) -> tuple[
    str,
    list[list[Any]],
    list[tuple[Image.Image, str]],
    list[tuple[Image.Image, str]],
    list[tuple[Image.Image, str]],
    list[tuple[Image.Image, str]],
    list[tuple[Image.Image, str]],
    dict[str, Any],
]:
    """Run the existing service and format artifacts for Gradio components."""
    started = perf_counter()
    report = _service().verify(
        _coerce_paths(references),
        _coerce_paths(queries),
        query_mode=InputMode(query_mode),
        signature_rate=signature_rate,
        signature_confidence=signature_confidence,
        signature_padding=signature_padding,
        signature_shrink=signature_shrink,
        query_top_cut=query_top_cut,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    return (
        _status_markdown(report, elapsed_ms),
        _comparison_rows(report),
        _candidate_gallery(report.reference_candidates),
        _model_input_gallery(report.reference_candidates),
        _candidate_gallery(report.query_candidates),
        _model_input_gallery(report.query_candidates),
        _document_gallery(report.query_bundles),
        report.to_dict(),
    )


def create_demo():
    gr = _load_gradio()
    service = _service()
    settings = service.settings
    with gr.Blocks(title="GPDS Signature Verifier Demo") as demo:
        gr.Markdown(
            "# GPDS Signature Verifier Demo\n"
            "Upload one or more cropped reference signatures and one or more query images. "
            "The demo uses the same ONNX GPDS pipeline as FastAPI."
        )
        with gr.Row():
            references = gr.File(
                label="Reference signatures (cropped)",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            queries = gr.File(
                label="Query documents or cropped signatures",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
        with gr.Row():
            query_mode = gr.Radio(
                choices=[mode.value for mode in InputMode],
                value=InputMode.DOCUMENT.value,
                label="Query mode",
            )
            signature_rate = gr.Slider(
                0,
                100,
                value=float(service.default_signature_rate),
                step=0.5,
                label="Signature rate (%)",
            )
        with gr.Accordion("Preprocessing / detector controls", open=False):
            with gr.Row():
                signature_confidence = gr.Slider(
                    0.01,
                    1.0,
                    value=float(settings.signature_confidence),
                    step=0.01,
                    label="Signature confidence",
                )
                query_top_cut = gr.Slider(
                    0.0,
                    0.60,
                    value=float(settings.query_top_cut),
                    step=0.01,
                    label="Query top cut",
                )
            with gr.Row():
                signature_padding = gr.Slider(
                    0.0,
                    0.20,
                    value=float(settings.signature_padding),
                    step=0.01,
                    label="Signature padding",
                )
                signature_shrink = gr.Slider(
                    0.0,
                    0.20,
                    value=float(settings.signature_shrink),
                    step=0.01,
                    label="Signature shrink",
                )
        run = gr.Button("Verify", variant="primary")
        status = gr.Markdown()
        comparisons = gr.Dataframe(
            headers=COMPARISON_HEADERS,
            datatype=["str", "str", "number", "number", "number", "str", "number"],
            interactive=False,
            label="Nearest-reference comparisons",
        )
        with gr.Tab("Extracted signatures"):
            with gr.Row():
                reference_crops = gr.Gallery(label="Reference cleaned crops", columns=3)
                query_crops = gr.Gallery(label="Query cleaned crops", columns=3)
        with gr.Tab("Model inputs"):
            with gr.Row():
                reference_inputs = gr.Gallery(label="Reference 224×224 model previews", columns=3)
                query_inputs = gr.Gallery(label="Query 224×224 model previews", columns=3)
        with gr.Tab("Document scan / detections"):
            document_artifacts = gr.Gallery(label="Query scanned documents and detection overlays", columns=2)
        with gr.Tab("Raw JSON"):
            raw_json = gr.JSON(label="Verification response")

        run.click(
            verify_with_gradio,
            inputs=[
                references,
                queries,
                query_mode,
                signature_rate,
                signature_confidence,
                signature_padding,
                signature_shrink,
                query_top_cut,
            ],
            outputs=[
                status,
                comparisons,
                reference_crops,
                reference_inputs,
                query_crops,
                query_inputs,
                document_artifacts,
                raw_json,
            ],
        )
    return demo


def main() -> None:
    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name=os.environ.get("SIGNATURE_GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("SIGNATURE_GRADIO_PORT", "7860")),
        show_error=True,
    )


if __name__ == "__main__":
    main()
