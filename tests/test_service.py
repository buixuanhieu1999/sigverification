from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image

from signature_verifier.domain import InputMode, SignatureCandidate, SourceBundle
from signature_verifier.service import SignatureVerificationService


def _candidate(role: str, name: str, number: int) -> SignatureCandidate:
    return SignatureCandidate(
        role=role,
        source_name=name,
        source_number=number,
        signature_number=1,
        image=Image.new("RGB", (16, 16), "white"),
        input_mode=InputMode.CROPPED_SIGNATURE,
    )


class _Extractor:
    def __init__(self, references, queries) -> None:
        self.references = references
        self.queries = queries

    def extract_many(self, files, *, role, **_options):
        candidates = self.references if role == "Reference" else self.queries
        return [SourceBundle(role=role, source_name=role, source_number=1, candidates=candidates)], []


class _Runtime:
    threshold = 0.5
    device = "cpu"
    session_id = 1
    epoch = 10
    metrics = {}

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def encode(self, candidates):
        return np.asarray([self.embeddings[item.source_name] for item in candidates], dtype=np.float32)

    def info(self):
        return {"active_device": "cpu", "model": "gpds_signature_encoder.onnx"}


def test_every_query_uses_its_nearest_reference(tmp_path) -> None:
    references = [_candidate("Reference", "ref-a.png", 1), _candidate("Reference", "ref-b.png", 2)]
    queries = [_candidate("Query", "query-a.png", 1), _candidate("Query", "query-b.png", 2)]
    service = SignatureVerificationService.__new__(SignatureVerificationService)
    service.settings = SimpleNamespace(
        signature_confidence=0.5,
        signature_padding=0.02,
        signature_shrink=0.04,
        query_top_cut=0.25,
        signature_rate=80.0,
    )
    service.extractor = _Extractor(references, queries)
    service.runtime = _Runtime(
        {
            "ref-a.png": [0.0, 0.0],
            "ref-b.png": [10.0, 0.0],
            "query-a.png": [0.1, 0.0],
            "query-b.png": [9.0, 0.0],
        }
    )
    service.lock = threading.Lock()

    input_paths = {
        name: tmp_path / name
        for name in ("ref-a.png", "ref-b.png", "query-a.png", "query-b.png")
    }
    for path in input_paths.values():
        path.touch()

    report = service.verify(
        [input_paths["ref-a.png"], input_paths["ref-b.png"]],
        [input_paths["query-a.png"], input_paths["query-b.png"]],
        query_mode=InputMode.CROPPED_SIGNATURE,
    )

    assert report.status == "mixed"
    assert report.rows[0].closest_reference.startswith("ref-a.png")
    assert np.isclose(report.rows[0].distance, 0.1)
    assert report.rows[0].matched is True
    row_payload = report.rows[0].to_dict()
    assert row_payload["similarity_percent"] == 96.0
    assert row_payload["signature_rate"] == 80.0
    assert report.rows[1].closest_reference.startswith("ref-b.png")
    assert report.rows[1].distance == 1.0
    assert report.rows[1].matched is False
    assert report.rows[1].to_dict()["similarity_percent"] == 60.0
    assert report.details["exact_pairs_evaluated"] == 4
    assert report.details["signature_rate"] == 80.0
    assert np.isclose(report.nearest_similarity_percent, 96.0)
