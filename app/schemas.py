from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MatrixSize = Annotated[int, Field(strict=True, ge=1)]
ImplementationName = Literal["cpu", "openmp", "cuda"]


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix_size: MatrixSize


class AdviceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=1_000)


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation: ImplementationName
    available: bool
    latency_ms: float | None = Field(default=None, ge=0)
    correct: bool | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_availability_fields(self) -> "Measurement":
        if self.available:
            if self.latency_ms is None or self.correct is None:
                raise ValueError(
                    "available measurements require latency_ms and correct"
                )
            if self.error is not None:
                raise ValueError("available measurements cannot contain an error")
        else:
            if self.latency_ms is not None or self.correct is not None:
                raise ValueError(
                    "unavailable measurements cannot contain latency or correctness"
                )
            if not self.error:
                raise ValueError("unavailable measurements require an error")
        return self


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix_size: MatrixSize
    data_type: Literal["float32"]
    transfer_included: bool
    cuda_context_warmup_excluded: bool
    gpu_name: str
    openmp_threads: int = Field(ge=1)
    measurements: list[Measurement] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def require_each_implementation_once(self) -> "BenchmarkResult":
        implementations = [item.implementation for item in self.measurements]
        if sorted(implementations) != ["cpu", "cuda", "openmp"]:
            raise ValueError("results must contain CPU, OpenMP, and CUDA exactly once")
        return self


class Analysis(BaseModel):
    fastest_implementation: ImplementationName | None
    speedups: dict[str, float]
    correctness_passed: bool
    complete_comparison: bool
    recommendation_allowed: bool
    notes: list[str]


class AdviceResponse(BaseModel):
    benchmark: BenchmarkResult
    analysis: Analysis
    explanation: str
    report: str
