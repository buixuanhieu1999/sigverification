"""Validated GPDS ONNX encoder runtime with dynamic batching."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .domain import SignatureCandidate
from .ort_session import create_session, run_with_cpu_fallback, runtime_info
from .preprocessing import SignatureTransform


def _metadata_value(metadata: dict[str, str], key: str, default: Any = None) -> Any:
    raw = metadata.get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GpdsEncoderRuntime:
    """Load one GPDS encoder and validate its production contract eagerly."""

    def __init__(
        self,
        model_path: Path,
        embedding_batch_size: int = 8,
        threshold_override: float | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.session, self.device = create_session(self.model_path)
        self.providers = self.session.get_providers()

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("GPDS encoder must have exactly one input and one output")
        input_node, output_node = inputs[0], outputs[0]
        if input_node.type != "tensor(float)" or list(input_node.shape[1:]) != [3, 224, 224]:
            raise ValueError(
                "GPDS encoder input must be float32 with shape [batch, 3, 224, 224]"
            )
        if output_node.type != "tensor(float)" or len(output_node.shape) != 2:
            raise ValueError("GPDS encoder output must be a rank-2 float32 embedding tensor")

        self.input_name = input_node.name
        self.output_name = output_node.name
        self.input_shape = list(input_node.shape)
        self.output_shape = list(output_node.shape)
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.model_name = str(
            _metadata_value(self.metadata, "model_name", "GPDS signature encoder")
        )
        embedded_threshold = _metadata_value(self.metadata, "threshold")
        if threshold_override is None and embedded_threshold is None:
            raise ValueError(
                "GPDS encoder has no threshold metadata. Set SIGNATURE_THRESHOLD explicitly."
            )
        threshold = threshold_override if threshold_override is not None else embedded_threshold
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Signature decision threshold must be a finite non-negative value")

        self.threshold_source = (
            "SIGNATURE_THRESHOLD environment override"
            if threshold_override is not None
            else "GPDS ONNX metadata"
        )
        self.session_id = _metadata_value(self.metadata, "checkpoint_session_id")
        self.epoch = _metadata_value(self.metadata, "checkpoint_epoch")
        self.metrics = _metadata_value(self.metadata, "metrics", {})
        self.model_sha256 = _sha256(self.model_path)
        self.transform = SignatureTransform()
        self.embedding_batch_size = max(1, int(embedding_batch_size))

    def encode(self, candidates: Sequence[SignatureCandidate]) -> np.ndarray:
        if not candidates:
            raise ValueError("Cannot encode an empty signature list")
        encoded: list[np.ndarray] = []
        for start in range(0, len(candidates), self.embedding_batch_size):
            chunk = candidates[start : start + self.embedding_batch_size]
            batch = np.stack([self.transform(candidate.image) for candidate in chunk])
            self.session, outputs = run_with_cpu_fallback(
                self.session,
                self.model_path,
                [self.output_name],
                {self.input_name: batch},
            )
            self.providers = self.session.get_providers()
            self.device = (
                "cuda"
                if self.providers and self.providers[0] == "CUDAExecutionProvider"
                else "cpu"
            )
            encoded.append(np.asarray(outputs[0], dtype=np.float32))
        return np.concatenate(encoded, axis=0)

    def info(self) -> dict[str, Any]:
        return {
            **runtime_info(),
            "active_device": self.device,
            "active_providers": self.providers,
            "model": self.model_path.name,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }
