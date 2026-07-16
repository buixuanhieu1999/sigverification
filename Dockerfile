# syntax=docker/dockerfile:1.7
ARG ORT_BACKEND=cpu
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /usr/local/bin/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./

FROM base AS dependencies-cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra onnx-cpu

FROM base AS dependencies-gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra onnx-gpu

FROM dependencies-${ORT_BACKEND} AS runtime
ARG ORT_BACKEND

COPY src ./src
COPY models ./models
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra onnx-${ORT_BACKEND} \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
ENV SIGNATURE_API_HOST=0.0.0.0 \
    SIGNATURE_API_PORT=8000 \
    SIGNATURE_ORT_PROVIDER=auto

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.environ.get('SIGNATURE_API_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health/ready', timeout=4)"

CMD ["signature-verifier-api"]
