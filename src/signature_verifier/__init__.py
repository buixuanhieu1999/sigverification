"""GPDS ONNX signature verification service."""

from .config import AppSettings
from .domain import InputMode, VerificationReport

__all__ = [
    "AppSettings",
    "InputMode",
    "SignatureVerificationService",
    "VerificationReport",
]


def __getattr__(name: str):
    if name == "SignatureVerificationService":
        from .service import SignatureVerificationService

        return SignatureVerificationService
    raise AttributeError(name)
