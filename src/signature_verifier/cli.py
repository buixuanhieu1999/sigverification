"""Small CLI smoke check for automation and future deployment health checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domain import InputMode
from .service import SignatureVerificationService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append")
    parser.add_argument("--query", type=Path, action="append")
    parser.add_argument("--query-mode", choices=[mode.value for mode in InputMode], default="cropped_signature")
    args = parser.parse_args()
    service = SignatureVerificationService()
    if not args.reference or not args.query:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "runtime": service.runtime.info(),
                    "signature_rate": service.default_signature_rate,
                },
                indent=2,
            )
        )
        return
    report = service.verify(
        args.reference,
        args.query,
        query_mode=InputMode(args.query_mode),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
