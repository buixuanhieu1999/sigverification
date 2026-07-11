"""Application configuration and project-relative ONNX model paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNATURE_CONFIDENCE = 0.50
DEFAULT_QUERY_TOP_CUT = 0.35
DEFAULT_SIGNATURE_RATE = 80.0


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class AppSettings:
    """Runtime paths and conservative production inference defaults."""

    encoder_model_path: Path
    document_model_path: Path
    signature_model_path: Path
    decision_threshold: float | None = None
    embedding_batch_size: int = 8
    max_input_edge: int = 4096
    signature_confidence: float = DEFAULT_SIGNATURE_CONFIDENCE
    signature_padding: float = 0.0
    signature_shrink: float = 0.0
    query_top_cut: float = DEFAULT_QUERY_TOP_CUT
    keep_largest_ink_groups: int = 3
    signature_rate: float = DEFAULT_SIGNATURE_RATE

    @classmethod
    def from_env(cls) -> "AppSettings":
        models = PROJECT_ROOT / "models"
        threshold_value = os.environ.get("SIGNATURE_THRESHOLD")
        return cls(
            encoder_model_path=_env_path(
                "SIGNATURE_ENCODER_MODEL", models / "gpds_signature_encoder.onnx"
            ),
            document_model_path=_env_path(
                "DOCUMENT_DETECTOR_MODEL", models / "document_detector.onnx"
            ),
            signature_model_path=_env_path(
                "SIGNATURE_DETECTOR_MODEL", models / "signature_detector.onnx"
            ),
            decision_threshold=(None if threshold_value is None else float(threshold_value)),
            embedding_batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "8")),
            signature_confidence=float(
                os.environ.get("SIGNATURE_CONFIDENCE", str(DEFAULT_SIGNATURE_CONFIDENCE))
            ),
            signature_padding=float(os.environ.get("SIGNATURE_PADDING", "0")),
            signature_shrink=float(os.environ.get("SIGNATURE_SHRINK", "0")),
            query_top_cut=float(os.environ.get("SIGNATURE_QUERY_TOP_CUT", str(DEFAULT_QUERY_TOP_CUT))),
            signature_rate=float(os.environ.get("SIGNATURE_RATE", str(DEFAULT_SIGNATURE_RATE))),
            keep_largest_ink_groups=int(
                os.environ.get("SIGNATURE_KEEP_LARGEST_INK_GROUPS", "3")
            ),
        )

    def with_overrides(self, **changes: object) -> "AppSettings":
        return replace(self, **{key: value for key, value in changes.items() if value is not None})

    def validate(self, *, require_detectors: bool = True) -> None:
        required = {"ResT ONNX encoder": self.encoder_model_path}
        if require_detectors:
            required.update(
                {
                    "document detector": self.document_model_path,
                    "signature detector": self.signature_model_path,
                }
            )
        missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing model asset(s):\n" + "\n".join(missing))
        invalid = [
            f"{label}: {path}"
            for label, path in required.items()
            if path.suffix.lower() != ".onnx"
        ]
        if invalid:
            raise ValueError("Only ONNX model assets are supported:\n" + "\n".join(invalid))
        if self.embedding_batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero")
        if self.keep_largest_ink_groups < 1:
            raise ValueError("SIGNATURE_KEEP_LARGEST_INK_GROUPS must be greater than zero")
        if not 0.01 <= self.signature_confidence <= 1.0:
            raise ValueError("SIGNATURE_CONFIDENCE must be between 0.01 and 1.0")
        if not 0.0 <= self.signature_padding <= 0.20:
            raise ValueError("SIGNATURE_PADDING must be between 0.0 and 0.20")
        if not 0.0 <= self.signature_shrink <= 0.20:
            raise ValueError("SIGNATURE_SHRINK must be between 0.0 and 0.20")
        if not 0.0 <= self.query_top_cut <= 0.60:
            raise ValueError("SIGNATURE_QUERY_TOP_CUT must be between 0.0 and 0.60")
        if not 0.0 <= self.signature_rate <= 100.0:
            raise ValueError("SIGNATURE_RATE must be between 0.0 and 100.0")
