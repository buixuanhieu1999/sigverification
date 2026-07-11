"""ONNX Runtime session creation with CUDA-first, CPU-safe fallback."""

from __future__ import annotations

import os
from pathlib import Path
import site
from typing import Any

import onnxruntime as ort


_DLL_DIRECTORY_HANDLES: list[Any] = []


def _prepare_cuda_libraries() -> None:
    """Make NVIDIA wheel sub-libraries discoverable, including lazy cuDNN DLLs."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    directories: list[Path] = []
    for root in site.getsitepackages():
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.is_dir():
            directories.extend(path for path in nvidia_root.glob("*/bin") if path.is_dir())
    for directory in directories:
        value = str(directory.resolve())
        if value not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(value))
        except OSError:
            continue


def provider_preference() -> str:
    value = os.environ.get("SIGNATURE_ORT_PROVIDER", "auto").strip().lower()
    if value not in {"auto", "cuda", "cpu"}:
        raise ValueError("SIGNATURE_ORT_PROVIDER must be one of: auto, cuda, cpu")
    return value


def create_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    """Create one session and report the provider that actually became active."""
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    preference = provider_preference()
    available = set(ort.get_available_providers())
    wants_cuda = preference != "cpu" and "CUDAExecutionProvider" in available

    if wants_cuda and hasattr(ort, "preload_dlls"):
        _prepare_cuda_libraries()
        try:
            # onnxruntime-gpu[cuda,cudnn] installs NVIDIA libraries in site-packages.
            ort.preload_dlls(directory="")
        except Exception:
            # Session creation below remains authoritative and can fall back to CPU.
            pass

    providers: list[str | tuple[str, dict[str, Any]]]
    if wants_cuda:
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        if preference == "cuda":
            raise RuntimeError(
                "CUDA was explicitly requested but CUDAExecutionProvider is unavailable. "
                "Install the onnx-gpu extra and verify NVIDIA/CUDA libraries."
            )
        providers = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
    except Exception:
        if preference == "cuda":
            raise
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    active = session.get_providers()
    device = "cuda" if active and active[0] == "CUDAExecutionProvider" else "cpu"
    return session, device


def run_with_cpu_fallback(
    session: ort.InferenceSession,
    model_path: Path,
    output_names: list[str],
    inputs: dict[str, Any],
) -> tuple[ort.InferenceSession, list[Any]]:
    """Run once; if auto CUDA fails, rebuild a CPU session and retry exactly once."""
    try:
        return session, session.run(output_names, inputs)
    except Exception:
        if provider_preference() != "auto" or session.get_providers() == ["CPUExecutionProvider"]:
            raise
        options = ort.SessionOptions()
        options.log_severity_level = 3
        cpu_session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        return cpu_session, cpu_session.run(output_names, inputs)


def runtime_info() -> dict[str, Any]:
    return {
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "provider_preference": provider_preference(),
    }
