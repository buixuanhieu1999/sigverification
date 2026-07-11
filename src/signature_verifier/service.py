"""Framework-neutral GPDS signature verification service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Sequence

import numpy as np

from .config import AppSettings
from .domain import ComparisonRow, InputMode, SourceBundle, VerificationReport
from .extractor import SignatureExtractor, normalize_paths
from .runtime import GpdsEncoderRuntime


def _finite_float(value: Any) -> float | None:
    if isinstance(value, int | float) and np.isfinite(float(value)):
        return float(value)
    return None


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


@dataclass(frozen=True)
class SimilarityScale:
    """Maps model L2 distance to a user-facing 0-100 similarity score."""

    positive_distance: float
    negative_distance: float
    source: str

    def percent(self, distance: float) -> float:
        span = self.negative_distance - self.positive_distance
        if span <= 0:
            return 0.0
        return _clamp_percent(((self.negative_distance - float(distance)) / span) * 100.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "positive_anchor_distance": round(self.positive_distance, 6),
            "negative_anchor_distance": round(self.negative_distance, 6),
        }


class SignatureVerificationService:
    """Owns model state and implements the complete verification use case.

    One service instance should be created per process. The lock prevents the
    document detector and ResT encoder from competing for the same GPU when a
    UI/server receives simultaneous requests.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings.from_env()
        self.settings.validate(require_detectors=True)
        self.extractor = SignatureExtractor(self.settings)
        self.runtime = GpdsEncoderRuntime(
            self.settings.encoder_model_path,
            embedding_batch_size=self.settings.embedding_batch_size,
            threshold_override=self.settings.decision_threshold,
        )
        self.lock = threading.Lock()

    @property
    def default_threshold(self) -> float:
        return self.runtime.threshold

    @property
    def default_signature_rate(self) -> float:
        return self.settings.signature_rate

    def similarity_scale(self) -> SimilarityScale:
        metrics = self.runtime.metrics if isinstance(self.runtime.metrics, dict) else {}
        positive = _finite_float(metrics.get("positive_mean_distance"))
        negative = _finite_float(metrics.get("negative_mean_distance"))
        if positive is not None and negative is not None and negative > positive:
            return SimilarityScale(
                positive_distance=positive,
                negative_distance=negative,
                source="model metrics: positive_mean_distance to negative_mean_distance",
            )

        # Internal fallback only. It keeps the legacy model threshold around 80%
        # on the user-facing scale when distance statistics are unavailable.
        positive = 0.0
        negative = max(self.default_threshold / 0.20, self.default_threshold + 1e-6)
        return SimilarityScale(
            positive_distance=positive,
            negative_distance=negative,
            source="fallback: internal model threshold maps near 80%",
        )

    @staticmethod
    def _bundle_metadata(bundles: Sequence[SourceBundle]) -> list[dict[str, Any]]:
        return [
            {
                "source": bundle.source_name,
                "source_number": bundle.source_number,
                "extracted_signatures": len(bundle.candidates),
                **bundle.metadata,
            }
            for bundle in bundles
        ]

    def verify(
        self,
        reference_files: Sequence[str | Path] | str | Path | None,
        query_files: Sequence[str | Path] | str | Path | None,
        *,
        query_mode: InputMode = InputMode.DOCUMENT,
        signature_rate: float | None = None,
        signature_confidence: float | None = None,
        signature_padding: float | None = None,
        signature_shrink: float | None = None,
        query_top_cut: float | None = None,
    ) -> VerificationReport:
        references = normalize_paths(reference_files)
        queries = normalize_paths(query_files)
        decision_rate = self.default_signature_rate if signature_rate is None else float(signature_rate)
        if not 0.0 <= decision_rate <= 100.0:
            return VerificationReport(
                status="failed",
                message="signature_rate must be between 0 and 100.",
                signature_rate=decision_rate,
            )
        if not references or not queries:
            return VerificationReport(
                status="failed",
                message="At least one reference and one query image are required.",
                signature_rate=decision_rate,
                details={"reference_uploads": len(references), "query_uploads": len(queries)},
            )

        confidence = (
            self.settings.signature_confidence
            if signature_confidence is None
            else float(signature_confidence)
        )
        padding = (
            self.settings.signature_padding
            if signature_padding is None
            else float(signature_padding)
        )
        shrink = (
            self.settings.signature_shrink
            if signature_shrink is None
            else float(signature_shrink)
        )
        top_cut = self.settings.query_top_cut if query_top_cut is None else float(query_top_cut)

        with self.lock:
            reference_bundles, reference_errors = self.extractor.extract_many(
                references,
                role="Reference",
                mode=InputMode.CROPPED_SIGNATURE,
                signature_confidence=confidence,
                signature_padding=padding,
                signature_shrink=shrink,
                query_top_cut=top_cut,
            )
            query_bundles, query_errors = self.extractor.extract_many(
                queries,
                role="Query",
                mode=query_mode,
                signature_confidence=confidence,
                signature_padding=padding,
                signature_shrink=shrink,
                query_top_cut=top_cut,
            )
            reference_candidates = [
                candidate for bundle in reference_bundles for candidate in bundle.candidates
            ]
            query_candidates = [
                candidate for bundle in query_bundles for candidate in bundle.candidates
            ]
            base_details = {
                "reference_sources": self._bundle_metadata(reference_bundles),
                "query_sources": self._bundle_metadata(query_bundles),
                "reference_upload_errors": reference_errors,
                "query_upload_errors": query_errors,
            }
            if not reference_candidates or not query_candidates:
                return VerificationReport(
                    status="failed",
                    message="No usable signatures were extracted on one side.",
                    signature_rate=decision_rate,
                    details=base_details,
                    reference_bundles=reference_bundles,
                    query_bundles=query_bundles,
                )

            reference_features = self.runtime.encode(reference_candidates)
            query_features = self.runtime.encode(query_candidates)
            distances = np.linalg.norm(
                query_features[:, None, :] - reference_features[None, :, :], axis=2
            )

        closest_indices = np.argmin(distances, axis=1)
        closest_distances = distances[np.arange(len(distances)), closest_indices]
        scale = self.similarity_scale()
        rows: list[ComparisonRow] = []
        match_count = 0
        for query_index, (distance, reference_index) in enumerate(
            zip(closest_distances, closest_indices)
        ):
            value = float(distance)
            similarity_percent = scale.percent(value)
            matched = similarity_percent >= decision_rate
            match_count += int(matched)
            rows.append(
                ComparisonRow(
                    query=query_candidates[query_index].short_label,
                    closest_reference=reference_candidates[int(reference_index)].short_label,
                    distance=value,
                    similarity_percent=similarity_percent,
                    signature_rate=decision_rate,
                    matched=matched,
                    hidden_reference_count=max(0, len(reference_candidates) - 1),
                )
            )

        if match_count == len(query_candidates):
            status = "all_match"
            message = "Every extracted query signature matches its nearest reference."
        elif match_count == 0:
            status = "no_match"
            message = "No extracted query signature matches its nearest reference."
        else:
            status = "mixed"
            message = "Some query signatures match and some require review."
        best_row = max(rows, key=lambda row: row.similarity_percent)
        nearest_distance = float(np.min(closest_distances))
        details = {
            **base_details,
            "decision_rule": "For each query: match when similarity_percent >= signature_rate.",
            "similarity_scale": "0-100; higher is closer to the nearest reference.",
            "signature_rate": round(decision_rate, 2),
            "similarity_calibration": scale.to_dict(),
            "reference_signatures": len(reference_candidates),
            "query_signatures": len(query_candidates),
            "exact_pairs_evaluated": len(reference_candidates) * len(query_candidates),
            "pairs_displayed": len(query_candidates),
            "closest_pairs": [row.to_dict() for row in rows],
            "runtime": self.runtime.info(),
            "device": self.runtime.device,
            "checkpoint_session": self.runtime.session_id,
            "checkpoint_epoch": self.runtime.epoch,
            "checkpoint_metrics": {
                key: round(float(value), 6)
                for key, value in self.runtime.metrics.items()
                if isinstance(value, int | float) and "threshold" not in str(key).lower()
            },
            "calibration_note": (
                "The similarity percentage is calibrated from GPDS distance statistics before "
                "document magic-scan/noise cleanup. Recalibrate SIGNATURE_RATE end-to-end on "
                "representative financial documents before using the decision as an automated control."
            ),
        }
        return VerificationReport(
            status=status,
            message=message,
            signature_rate=decision_rate,
            rows=rows,
            nearest_distance=nearest_distance,
            nearest_similarity_percent=best_row.similarity_percent,
            details=details,
            reference_bundles=reference_bundles,
            query_bundles=query_bundles,
        )
