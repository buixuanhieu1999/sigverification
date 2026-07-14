from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from signature_verifier.api.app import UploadLimits, create_app
from signature_verifier.domain import ComparisonRow, InputMode, VerificationReport


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 20), "white").save(output, format="PNG")
    return output.getvalue()


class _Node:
    def __init__(self, name: str, shape: list[str | int]) -> None:
        self.name = name
        self.shape = shape


class _Session:
    def get_inputs(self):
        return [_Node("images", ["batch", 3, 224, 224])]

    def get_outputs(self):
        return [_Node("embeddings", ["batch", 1280])]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={"model_name": '"ResT GPDS encoder"'})


class _Runtime:
    model_path = Path("models/gpds_signature_encoder.onnx")
    device = "cpu"
    providers = ["CPUExecutionProvider"]
    session_id = 1
    epoch = 10
    session = _Session()
    model_sha256 = "a" * 64


class _FakeService:
    default_threshold = 0.3709427153002471
    default_signature_rate = 80.0
    runtime = _Runtime()

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def verify(self, references, queries, **options):
        self.calls.append(
            {
                "references": list(references),
                "queries": list(queries),
                "reference_bytes": [path.read_bytes() for path in references],
                "query_bytes": [path.read_bytes() for path in queries],
                "thread": threading.get_ident(),
                "options": options,
            }
        )
        return VerificationReport(
            status="all_match",
            message="match",
            signature_rate=options.get("signature_rate", self.default_signature_rate),
            nearest_distance=0.2,
            nearest_similarity_percent=92.0,
            rows=[
                ComparisonRow(
                    query="query.png",
                    closest_reference="reference.png",
                    distance=0.2,
                    similarity_percent=92.0,
                    signature_rate=options.get("signature_rate", self.default_signature_rate),
                    matched=True,
                    hidden_reference_count=max(0, len(references) - 1),
                )
            ],
            details={
                "runtime": {
                    "model": r"C:\private\models\gpds_signature_encoder.onnx",
                    "active_device": "cpu",
                }
            },
        )


def test_health_model_and_multipart_verification_use_one_service_instance() -> None:
    service = _FakeService()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return service

    application = create_app(factory)
    image = _png_bytes()
    with TestClient(application) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["providers"] == ["CPUExecutionProvider"]

        model = client.get("/v1/model")
        assert model.status_code == 200
        assert model.json()["name"] == "ResT GPDS encoder"
        assert model.json()["signature_rate"] == 80.0
        assert model.json()["input_shape"] == ["batch", 3, 224, 224]
        assert model.json()["output_shape"] == ["batch", 1280]
        assert "threshold" not in model.json()

        result = client.post(
            "/v1/verify",
            files=[
                ("references", ("reference.png", image, "image/png")),
                ("references", ("reference-2.png", image, "image/png")),
                ("queries", ("query.png", image, "image/png")),
            ],
            data={"query_mode": "cropped_signature"},
        )

    assert factory_calls == 1
    assert result.status_code == 200
    assert result.headers["x-request-id"] == result.json()["request_id"]
    assert result.json()["status"] == "all_match"
    assert result.json()["comparisons"][0]["prediction"] == "match"
    assert result.json()["comparisons"][0]["similarity_percent"] == 92.0
    assert result.json()["signature_rate"] == 80.0
    assert result.json()["runtime"]["model"] == "gpds_signature_encoder.onnx"
    assert "model" not in result.json()["details"]["runtime"]
    assert len(service.calls) == 1
    assert service.calls[0]["options"]["signature_rate"] == 80.0
    assert service.calls[0]["reference_bytes"] == [image, image]
    assert service.calls[0]["query_bytes"] == [image]
    assert service.calls[0]["options"]["query_mode"] is InputMode.CROPPED_SIGNATURE
    assert all(not path.exists() for path in service.calls[0]["references"])
    assert all(not path.exists() for path in service.calls[0]["queries"])


def test_verify_openapi_schema_renders_swagger_file_uploads() -> None:
    application = create_app(lambda: _FakeService())

    with TestClient(application) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    request_body_ref = schema["paths"]["/v1/verify"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["$ref"]
    body_schema_name = request_body_ref.rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][body_schema_name]["properties"]
    assert properties["references"]["items"] == {"type": "string", "format": "binary"}
    assert properties["queries"]["items"] == {"type": "string", "format": "binary"}
    assert "threshold" not in properties
    assert "threshold_percent" not in properties
    assert properties["signature_rate"]["default"] == 80.0
    assert properties["signature_rate"]["minimum"] == 0.0
    assert properties["signature_rate"]["maximum"] == 100.0
    assert properties["signature_confidence"]["default"] == 0.5
    assert properties["signature_padding"]["default"] == 0.0
    assert properties["signature_shrink"]["default"] == 0.0
    assert properties["query_top_cut"]["default"] == 0.19


def test_signature_rate_is_passed_to_service() -> None:
    service = _FakeService()
    application = create_app(lambda: service)
    image = _png_bytes()

    with TestClient(application) as client:
        response = client.post(
            "/v1/verify",
            files=[
                ("references", ("reference.png", image, "image/png")),
                ("queries", ("query.png", image, "image/png")),
            ],
            data={"signature_rate": "72.5"},
        )

    assert response.status_code == 200
    assert service.calls[0]["options"]["signature_rate"] == 72.5


def test_upload_size_limit_returns_typed_error_and_skips_inference() -> None:
    service = _FakeService()
    application = create_app(
        lambda: service,
        upload_limits=UploadLimits(
            max_files_per_side=2,
            max_bytes_per_file=10,
            max_total_bytes=20,
            max_image_pixels=1_000,
        ),
    )
    image = _png_bytes()
    with TestClient(application) as client:
        response = client.post(
            "/v1/verify",
            files=[
                ("references", ("reference.png", image, "image/png")),
                ("queries", ("query.png", image, "image/png")),
            ],
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert response.json()["error"]["request_id"]
    assert not service.calls


def test_unsupported_extension_is_rejected_before_inference() -> None:
    service = _FakeService()
    application = create_app(lambda: service)
    with TestClient(application) as client:
        response = client.post(
            "/v1/verify",
            files=[
                ("references", ("reference.txt", b"not an image", "text/plain")),
                ("queries", ("query.png", _png_bytes(), "image/png")),
            ],
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_image_type"
    assert not service.calls


def test_broken_image_payload_is_rejected_before_inference() -> None:
    service = _FakeService()
    application = create_app(lambda: service)
    with TestClient(application) as client:
        response = client.post(
            "/v1/verify",
            files=[
                ("references", ("reference.png", b"not a png", "image/png")),
                ("queries", ("query.png", _png_bytes(), "image/png")),
            ],
        )

    assert response.status_code == 422
    assert response.headers["x-request-id"] == response.json()["error"]["request_id"]
    assert response.json()["error"]["code"] == "invalid_image"
    assert not service.calls
