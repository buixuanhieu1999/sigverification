"""Framework-neutral objects used by the verification core and HTTP API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Literal

from PIL import Image


def json_safe(value: Any) -> Any:
    """Recursively normalize values for strict JSON encoders."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # NumPy scalar values expose item(), but NumPy remains optional to this module.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


class InputMode(str, Enum):
    DOCUMENT = "document"
    CROPPED_SIGNATURE = "cropped_signature"


@dataclass
class SignatureCandidate:
    role: str
    source_name: str
    source_number: int
    signature_number: int
    image: Image.Image
    detection: dict[str, Any] | None = None
    input_mode: InputMode = InputMode.DOCUMENT

    @property
    def confidence(self) -> float | None:
        value = None if self.detection is None else self.detection.get("confidence")
        return float(value) if isinstance(value, int | float) else None

    @property
    def label(self) -> str:
        suffix = (
            "magic-scanned and cleaned crop"
            if self.confidence is None
            else f"detector {self.confidence:.2f}"
        )
        return (
            f"{self.role} {self.source_number}: {self.source_name} · "
            f"signature {self.signature_number} · {suffix}"
        )

    @property
    def short_label(self) -> str:
        return f"{self.source_name} · signature {self.signature_number}"


@dataclass
class SourceBundle:
    role: str
    source_name: str
    source_number: int
    candidates: list[SignatureCandidate] = field(default_factory=list)
    warped_document: Image.Image | None = None
    detection_overlay: Image.Image | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonRow:
    query: str
    closest_reference: str
    distance: float
    similarity_percent: float
    signature_rate: float
    matched: bool
    hidden_reference_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "closest_reference": self.closest_reference,
            "distance": round(self.distance, 6),
            "similarity_percent": round(self.similarity_percent, 2),
            "signature_rate": round(self.signature_rate, 2),
            "prediction": "match" if self.matched else "no_match",
            "hidden_reference_count": self.hidden_reference_count,
        }


@dataclass
class VerificationReport:
    status: Literal["all_match", "no_match", "mixed", "failed"]
    message: str
    signature_rate: float
    rows: list[ComparisonRow] = field(default_factory=list)
    nearest_distance: float | None = None
    nearest_similarity_percent: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    reference_bundles: list[SourceBundle] = field(default_factory=list)
    query_bundles: list[SourceBundle] = field(default_factory=list)

    @property
    def reference_candidates(self) -> list[SignatureCandidate]:
        return [candidate for bundle in self.reference_bundles for candidate in bundle.candidates]

    @property
    def query_candidates(self) -> list[SignatureCandidate]:
        return [candidate for bundle in self.query_bundles for candidate in bundle.candidates]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe business result; image artifacts deliberately stay out."""
        return json_safe({
            "status": self.status,
            "message": self.message,
            "signature_rate": round(self.signature_rate, 2),
            "nearest_distance": (
                None if self.nearest_distance is None else round(self.nearest_distance, 6)
            ),
            "nearest_similarity_percent": (
                None
                if self.nearest_similarity_percent is None
                else round(self.nearest_similarity_percent, 2)
            ),
            "comparisons": [row.to_dict() for row in self.rows],
            "details": self.details,
        })
