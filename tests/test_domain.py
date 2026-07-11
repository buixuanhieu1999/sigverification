import json
from pathlib import Path

import numpy as np

from signature_verifier.domain import VerificationReport


def test_report_is_safe_for_strict_json_encoding() -> None:
    report = VerificationReport(
        status="all_match",
        message="ok",
        signature_rate=80.0,
        details={
            "names": {0: "signature", 73: "book"},
            "model": Path("models/example.onnx"),
            "score": np.float32(0.8),
        },
    )
    payload = report.to_dict()
    assert json.dumps(payload, allow_nan=False)
    assert payload["details"]["names"] == {"0": "signature", "73": "book"}
