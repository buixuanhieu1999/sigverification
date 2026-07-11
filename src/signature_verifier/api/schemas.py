"""Stable HTTP response schemas for the signature verification API."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class LiveHealthResponse(BaseModel):
    status: Literal["alive"] = "alive"


class ReadyHealthResponse(BaseModel):
    status: Literal["ready"] = "ready"
    device: str
    providers: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    name: str
    sha256: str
    signature_rate: float
    checkpoint_session: str | int | None = None
    checkpoint_epoch: str | int | None = None
    device: str
    providers: list[str] = Field(default_factory=list)
    input_name: str | None = None
    input_shape: list[str | int | None] = Field(default_factory=list)
    output_name: str | None = None
    output_shape: list[str | int | None] = Field(default_factory=list)


class RuntimeSummary(BaseModel):
    device: str
    providers: list[str] = Field(default_factory=list)
    model: str
    model_sha256: str
    checkpoint_session: str | int | None = None
    checkpoint_epoch: str | int | None = None


class ComparisonResponse(BaseModel):
    query: str
    closest_reference: str
    distance: float
    similarity_percent: float
    signature_rate: float
    prediction: Literal["match", "no_match"]
    hidden_reference_count: int = Field(ge=0)


class VerificationResponse(BaseModel):
    request_id: UUID
    status: Literal["all_match", "no_match", "mixed", "failed"]
    message: str
    signature_rate: float
    nearest_distance: float | None = None
    nearest_similarity_percent: float | None = None
    comparisons: list[ComparisonResponse] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    runtime: RuntimeSummary
    processing_ms: float = Field(ge=0)


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: UUID | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


