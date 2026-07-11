# GPDS Signature Verifier

An ONNX-only service that compares signatures on financial documents. It includes
the production ONNX models, a FastAPI API, a small optional Gradio demo, Docker
configuration, and regression tests. Training and model-export code are deliberately
out of scope.

## What it does

1. Prepares cropped reference signatures with the GPDS preprocessing pipeline.
2. For document queries, detects and warps the page, detects signatures, and cleans
   each crop. Cropped-signature queries skip document detection.
3. Encodes every signature with the GPDS ONNX model and selects the closest reference
   by L2 distance.
4. Reports a match when the derived similarity is at least `SIGNATURE_RATE` (80 by
   default). This score is a model signal, not a legal-confidence probability.

## Project layout

```text
models/                    Production ONNX assets and checksum manifest
src/signature_verifier/    Framework-neutral verification core and interfaces
  api/                     FastAPI application and request schemas
  scanner/                 Document and signature detection
tests/                     Unit and model-contract regression tests
Dockerfile                 CPU/GPU runtime image
compose*.yaml              Docker Compose configurations
```

Generated inspection images belong in `outputs/`; the directory is ignored and is
not part of the source repository.

## Run locally

Install the CPU runtime and start the API:

```bash
uv sync --frozen --extra onnx-cpu
uv run signature-verifier-api
```

The API is available at `http://localhost:8000` by default. OpenAPI documentation is at
`/docs`.

For a GPU runtime, install the GPU extra instead and set the provider when needed:

```bash
uv sync --frozen --extra onnx-gpu
SIGNATURE_ORT_PROVIDER=auto uv run signature-verifier-api
```

On PowerShell, set the environment variable first:

```powershell
$env:SIGNATURE_ORT_PROVIDER = "auto"
uv run signature-verifier-api
```

The optional local demo uses the same service:

```bash
uv run --extra onnx-cpu --extra demo signature-verifier-demo
```

## Run with Docker

CPU is the default:

```bash
docker compose up --build
```

For NVIDIA GPUs (with NVIDIA Container Toolkit installed):

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Use one Uvicorn worker per container: each worker loads the models independently.
Scale with more containers when required.

## API

- `GET /health/live` — process liveness.
- `GET /health/ready` — model readiness.
- `GET /v1/model` — active model metadata and SHA-256.
- `POST /v1/verify` — multipart verification request.

Example:

```bash
curl -X POST http://localhost:8000/v1/verify \
  -F "references=@reference-1.png" \
  -F "references=@reference-2.png" \
  -F "queries=@receipt.jpg" \
  -F "query_mode=document"
```

Use `query_mode=cropped_signature` when each query is already a signature crop.

## Configuration

Copy `.env.example` to `.env` only when overriding defaults. The most useful settings
are:

- Model paths: `SIGNATURE_ENCODER_MODEL`, `DOCUMENT_DETECTOR_MODEL`,
  `SIGNATURE_DETECTOR_MODEL`
- Runtime: `SIGNATURE_ORT_PROVIDER=auto|cuda|cpu`, `EMBEDDING_BATCH_SIZE`
- Decision and detection tuning: `SIGNATURE_RATE`, `SIGNATURE_CONFIDENCE`,
  `SIGNATURE_PADDING`, `SIGNATURE_SHRINK`, `SIGNATURE_QUERY_TOP_CUT`
- Upload limits: `SIGNATURE_MAX_FILES_PER_SIDE`, `SIGNATURE_MAX_UPLOAD_MB`,
  `SIGNATURE_MAX_TOTAL_UPLOAD_MB`, `SIGNATURE_MAX_IMAGE_MEGAPIXELS`

## Upgrade a model safely

1. Replace the intended `.onnx` file under `models/` (or point the matching setting
   at a mounted file).
2. Update `models/manifest.json` with the new file size and SHA-256.
3. Run the test suite and check `/health/ready` and `/v1/model` before deployment.

The encoder must accept `float32 [batch, 3, 224, 224]` and return
`float32 [batch, embedding_dimension]`. It should provide a `threshold` metadata
value, or `SIGNATURE_THRESHOLD` must be set.

## Development checks

```bash
uv run --no-sync pytest
uv run --no-sync ruff check src tests
```

The GPDS encoder is stored with Git LFS. Install Git LFS before cloning this repository
so the model is downloaded instead of its lightweight pointer file.

## Deployment notes

The service does not provide authentication or TLS. Place it behind a gateway or
reverse proxy that supplies HTTPS, authentication, rate limits, and audit logging.
The detector metadata reports an AGPL-3.0 Ultralytics license; review that and the
GPDS model/dataset rights before commercial or closed-source distribution.
