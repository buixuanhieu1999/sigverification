"""FastAPI application for GPDS signature verification."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Annotated, Any, Callable, Sequence
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from ..config import (
    DEFAULT_QUERY_TOP_CUT,
    DEFAULT_SIGNATURE_CONFIDENCE,
    DEFAULT_SIGNATURE_RATE,
    AppSettings,
)
from ..domain import InputMode, VerificationReport
from ..service import SignatureVerificationService
from .schemas import (
    ComparisonResponse,
    ErrorResponse,
    LiveHealthResponse,
    ModelInfoResponse,
    ReadyHealthResponse,
    RuntimeSummary,
    VerificationResponse,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
ServiceFactory = Callable[[], SignatureVerificationService]
VERIFY_BODY_SCHEMA = "Body_verify_v1_verify_post"


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class UploadLimits:
    """Limits applied while streaming multipart uploads to request-local storage."""

    max_files_per_side: int = 20
    max_bytes_per_file: int = 20 * 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    @classmethod
    def from_env(cls) -> "UploadLimits":
        return cls(
            max_files_per_side=_positive_env_int("SIGNATURE_MAX_FILES_PER_SIDE", 20),
            max_bytes_per_file=(
                _positive_env_int("SIGNATURE_MAX_UPLOAD_MB", 20) * 1024 * 1024
            ),
            max_total_bytes=(
                _positive_env_int("SIGNATURE_MAX_TOTAL_UPLOAD_MB", 100) * 1024 * 1024
            ),
            max_image_pixels=(
                _positive_env_int("SIGNATURE_MAX_IMAGE_MEGAPIXELS", 40) * 1_000_000
            ),
        )


@dataclass
class _UploadBudget:
    total_bytes: int = 0


class _APIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        request_id: UUID | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id


def _default_service_factory() -> SignatureVerificationService:
    settings = AppSettings.from_env()
    # Fail fast during container startup instead of waiting for the first
    # document request to discover a missing detector.
    settings.validate(require_detectors=True)
    return SignatureVerificationService(settings)


def _safe_filename(filename: str | None, index: int) -> str:
    leaf = Path(filename or f"upload-{index}.png").name
    leaf = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._")
    return (leaf or f"upload-{index}.png")[-180:]


async def _persist_uploads(
    uploads: Sequence[UploadFile],
    directory: Path,
    *,
    side: str,
    limits: UploadLimits,
    budget: _UploadBudget,
    request_id: UUID,
) -> list[Path]:
    if not uploads:
        raise _APIError(422, "missing_uploads", f"At least one {side} image is required.", request_id)
    if len(uploads) > limits.max_files_per_side:
        raise _APIError(
            413,
            "too_many_files",
            f"{side} accepts at most {limits.max_files_per_side} images.",
            request_id,
        )

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, upload in enumerate(uploads, start=1):
        filename = _safe_filename(upload.filename, index)
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise _APIError(
                415,
                "unsupported_image_type",
                f"Unsupported {side} file type: {suffix or 'no extension'}.",
                request_id,
            )
        content_type = (upload.content_type or "").lower()
        if content_type and not (
            content_type.startswith("image/") or content_type == "application/octet-stream"
        ):
            raise _APIError(
                415,
                "unsupported_media_type",
                f"Unsupported {side} content type: {content_type}.",
                request_id,
            )

        destination = directory / f"{index:03d}-{filename}"
        file_bytes = 0
        with destination.open("wb") as output:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                file_bytes += len(chunk)
                budget.total_bytes += len(chunk)
                if file_bytes > limits.max_bytes_per_file:
                    raise _APIError(
                        413,
                        "file_too_large",
                        f"Each image must be at most {limits.max_bytes_per_file} bytes.",
                        request_id,
                    )
                if budget.total_bytes > limits.max_total_bytes:
                    raise _APIError(
                        413,
                        "request_too_large",
                        f"Combined uploads must be at most {limits.max_total_bytes} bytes.",
                        request_id,
                    )
                output.write(chunk)
        if file_bytes == 0:
            raise _APIError(422, "empty_upload", f"{filename} is empty.", request_id)
        try:
            with Image.open(destination) as image:
                width, height = image.size
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise _APIError(
                422, "invalid_image", f"{filename} is not a readable image.", request_id
            ) from exc
        if width * height > limits.max_image_pixels:
            raise _APIError(
                413,
                "image_too_large",
                f"Decoded image must have at most {limits.max_image_pixels} pixels.",
                request_id,
            )
        paths.append(destination)
    return paths


async def _close_uploads(uploads: Sequence[UploadFile]) -> None:
    for upload in uploads:
        await upload.close()


def _metadata_value(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _node_contract(node: Any | None) -> tuple[str | None, list[str | int | None]]:
    if node is None:
        return None, []
    return getattr(node, "name", None), list(getattr(node, "shape", []) or [])


def _model_info(service: SignatureVerificationService) -> ModelInfoResponse:
    runtime = service.runtime
    session = getattr(runtime, "session", None)
    inputs = session.get_inputs() if session is not None else []
    outputs = session.get_outputs() if session is not None else []
    input_name, input_shape = _node_contract(inputs[0] if inputs else None)
    output_name, output_shape = _node_contract(outputs[0] if outputs else None)

    metadata: dict[str, str] = {}
    if session is not None:
        model_meta = session.get_modelmeta()
        metadata = dict(getattr(model_meta, "custom_metadata_map", {}) or {})
    model_path = Path(getattr(runtime, "model_path", "gpds_signature_encoder.onnx"))
    model_name = _metadata_value(metadata.get("model_name")) or model_path.stem
    return ModelInfoResponse(
        name=str(model_name),
        sha256=str(getattr(runtime, "model_sha256", "unknown")),
        signature_rate=float(service.default_signature_rate),
        checkpoint_session=getattr(runtime, "session_id", None),
        checkpoint_epoch=getattr(runtime, "epoch", None),
        device=str(getattr(runtime, "device", "unknown")),
        providers=list(getattr(runtime, "providers", []) or []),
        input_name=input_name,
        input_shape=input_shape,
        output_name=output_name,
        output_shape=output_shape,
    )


def _runtime_summary(service: SignatureVerificationService) -> RuntimeSummary:
    runtime = service.runtime
    model_path = Path(getattr(runtime, "model_path", "gpds_signature_encoder.onnx"))
    return RuntimeSummary(
        device=str(getattr(runtime, "device", "unknown")),
        providers=list(getattr(runtime, "providers", []) or []),
        model=model_path.name,
        model_sha256=str(getattr(runtime, "model_sha256", "unknown")),
        checkpoint_session=getattr(runtime, "session_id", None),
        checkpoint_epoch=getattr(runtime, "epoch", None),
    )


def _force_swagger_file_upload_schema(openapi_schema: dict[str, Any]) -> None:
    """Make multipart file arrays render as file pickers in Swagger UI."""
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    verify_body = schemas.get(VERIFY_BODY_SCHEMA)
    if not isinstance(verify_body, dict):
        return

    properties = verify_body.get("properties", {})
    if not isinstance(properties, dict):
        return

    for field_name in ("references", "queries"):
        field = properties.get(field_name)
        if not isinstance(field, dict):
            continue
        field["type"] = "array"
        field["items"] = {"type": "string", "format": "binary"}


def _public_details(value: Any) -> Any:
    """Remove host paths while preserving useful detector/runtime metadata."""
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key == "runtime" and isinstance(item, dict):
                item = {part: val for part, val in item.items() if part != "model"}
            if isinstance(item, str) and (key == "model" or key.endswith("_path")):
                item = item.replace("\\", "/").rsplit("/", 1)[-1]
            output[str(key)] = _public_details(item)
        return output
    if isinstance(value, list):
        return [_public_details(item) for item in value]
    return value


def _verification_response(
    service: SignatureVerificationService,
    report: VerificationReport,
    *,
    request_id: UUID,
    processing_ms: float,
) -> VerificationResponse:
    payload = report.to_dict()
    comparisons = [ComparisonResponse(**row) for row in payload.get("comparisons", [])]
    return VerificationResponse(
        request_id=request_id,
        status=payload["status"],
        message=payload["message"],
        signature_rate=float(payload["signature_rate"]),
        nearest_distance=payload.get("nearest_distance"),
        nearest_similarity_percent=payload.get("nearest_similarity_percent"),
        comparisons=comparisons,
        details=_public_details(payload.get("details", {})),
        runtime=_runtime_summary(service),
        processing_ms=max(0.0, processing_ms),
    )


def create_app(
    service_factory: ServiceFactory | None = None,
    *,
    upload_limits: UploadLimits | None = None,
) -> FastAPI:
    """Create an app; injectable factories keep HTTP tests independent of ONNX assets."""

    factory = service_factory or _default_service_factory
    limits = upload_limits or UploadLimits.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        application.state.service = await run_in_threadpool(factory)
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False

    application = FastAPI(
        title="GPDS Signature Verifier",
        version="1.0.0",
        lifespan=lifespan,
    )

    def custom_openapi() -> dict[str, Any]:
        if application.openapi_schema:
            return application.openapi_schema
        openapi_schema = get_openapi(
            title=application.title,
            version=application.version,
            description=application.description,
            routes=application.routes,
        )
        _force_swagger_file_upload_schema(openapi_schema)
        application.openapi_schema = openapi_schema
        return application.openapi_schema

    application.openapi = custom_openapi  # type: ignore[method-assign]

    @application.exception_handler(_APIError)
    async def api_error_handler(_: Request, exc: _APIError) -> JSONResponse:
        response = JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": None if exc.request_id is None else str(exc.request_id),
                }
            },
        )
        if exc.request_id is not None:
            response.headers["X-Request-ID"] = str(exc.request_id)
        return response

    def service_from(request: Request) -> SignatureVerificationService:
        service = getattr(request.app.state, "service", None)
        if not getattr(request.app.state, "ready", False) or service is None:
            raise _APIError(503, "service_not_ready", "Inference service is not ready.")
        return service

    @application.get("/health/live", response_model=LiveHealthResponse, tags=["health"])
    async def health_live() -> LiveHealthResponse:
        return LiveHealthResponse()

    @application.get(
        "/health/ready",
        response_model=ReadyHealthResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["health"],
    )
    async def health_ready(request: Request) -> ReadyHealthResponse:
        service = service_from(request)
        runtime = service.runtime
        return ReadyHealthResponse(
            device=str(getattr(runtime, "device", "unknown")),
            providers=list(getattr(runtime, "providers", []) or []),
        )

    @application.get(
        "/v1/model",
        response_model=ModelInfoResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["model"],
    )
    async def model_info(request: Request) -> ModelInfoResponse:
        return _model_info(service_from(request))

    @application.post(
        "/v1/verify",
        response_model=VerificationResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["verification"],
    )
    async def verify(
        request: Request,
        response: Response,
        references: Annotated[list[UploadFile], File(description="Cropped reference signatures")],
        queries: Annotated[list[UploadFile], File(description="Query documents or signatures")],
        query_mode: Annotated[InputMode, Form()] = InputMode.DOCUMENT,
        signature_rate: Annotated[
            float,
            Form(
                ge=0.0,
                le=100.0,
                description=(
                    "Required match percentage on a 0-100 scale. "
                    "The query matches when similarity_percent >= signature_rate."
                ),
            ),
        ] = DEFAULT_SIGNATURE_RATE,
        signature_confidence: Annotated[
            float,
            Form(
                ge=0.01,
                le=1.0,
                description="Signature detector confidence. Default follows detector model default.",
            ),
        ] = DEFAULT_SIGNATURE_CONFIDENCE,
        signature_padding: Annotated[float, Form(ge=0.0, le=0.20)] = 0.0,
        signature_shrink: Annotated[float, Form(ge=0.0, le=0.20)] = 0.0,
        query_top_cut: Annotated[float, Form(ge=0.0, le=0.60)] = DEFAULT_QUERY_TOP_CUT,
    ) -> VerificationResponse:
        service = service_from(request)
        request_id = uuid4()
        response.headers["X-Request-ID"] = str(request_id)
        started = perf_counter()
        budget = _UploadBudget()
        try:
            with TemporaryDirectory(prefix="signature-verifier-") as temporary_directory:
                root = Path(temporary_directory)
                reference_paths = await _persist_uploads(
                    references,
                    root / "references",
                    side="reference",
                    limits=limits,
                    budget=budget,
                    request_id=request_id,
                )
                query_paths = await _persist_uploads(
                    queries,
                    root / "queries",
                    side="query",
                    limits=limits,
                    budget=budget,
                    request_id=request_id,
                )
                operation = partial(
                    service.verify,
                    reference_paths,
                    query_paths,
                    query_mode=query_mode,
                    signature_rate=signature_rate,
                    signature_confidence=signature_confidence,
                    signature_padding=signature_padding,
                    signature_shrink=signature_shrink,
                    query_top_cut=query_top_cut,
                )
                report = await run_in_threadpool(operation)
                return _verification_response(
                    service,
                    report,
                    request_id=request_id,
                    processing_ms=(perf_counter() - started) * 1000.0,
                )
        finally:
            await _close_uploads([*references, *queries])

    return application


app = create_app()


def main() -> None:
    """Run the development server; Docker invokes Uvicorn directly."""
    import uvicorn

    uvicorn.run(
        "signature_verifier.api.app:app",
        host=os.environ.get("SIGNATURE_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("SIGNATURE_API_PORT", "8000")),
        workers=1,
    )
