import hashlib
import json
from pathlib import Path
import sys

import onnxruntime as ort

from signature_verifier.config import AppSettings, PROJECT_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_production_import_path_has_no_training_or_demo_frameworks() -> None:
    import signature_verifier.service  # noqa: F401

    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules
    assert "gradio" not in sys.modules


def test_gpds_encoder_contract_and_threshold_metadata() -> None:
    settings = AppSettings.from_env()
    session = ort.InferenceSession(
        str(settings.encoder_model_path), providers=["CPUExecutionProvider"]
    )
    input_node = session.get_inputs()[0]
    output_node = session.get_outputs()[0]
    metadata = session.get_modelmeta().custom_metadata_map
    assert input_node.name == "images"
    assert input_node.shape == ["batch", 3, 224, 224]
    assert input_node.type == "tensor(float)"
    assert output_node.name == "embeddings"
    assert output_node.shape == ["batch", 1280]
    assert output_node.type == "tensor(float)"
    assert json.loads(metadata["model_name"]) == "ResT GPDS encoder"
    assert float(json.loads(metadata["threshold"])) == 0.3709427153002471


def test_manifest_hashes_match_all_production_models() -> None:
    manifest = json.loads((PROJECT_ROOT / "models" / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert {item["role"] for item in artifacts} == {
        "signature_encoder",
        "document_detector",
        "signature_detector",
    }
    for artifact in artifacts:
        path = PROJECT_ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_detector_contracts_have_dynamic_spatial_shape() -> None:
    settings = AppSettings.from_env()
    for path in (settings.document_model_path, settings.signature_model_path):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        node = session.get_inputs()[0]
        assert node.name == "images"
        assert node.shape[1] == 3
        assert isinstance(node.shape[2], str)
        assert isinstance(node.shape[3], str)


def test_serving_project_contains_no_checkpoint_files() -> None:
    assert not list(PROJECT_ROOT.rglob("*.pt"))
