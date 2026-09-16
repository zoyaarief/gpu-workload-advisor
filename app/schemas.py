from __future__ import annotations

from statistics import median
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MatrixSize = Annotated[int, Field(strict=True, ge=1)]
ImplementationName = Literal["cpu", "openmp", "cuda", "cuda_multi_gpu"]
TelemetryPhase = Literal["idle", "cuda", "cuda_multi_gpu"]

IMPLEMENTATIONS: tuple[ImplementationName, ...] = (
    "cpu",
    "openmp",
    "cuda",
    "cuda_multi_gpu",
)
# Native output prints six decimals, so a recomputed median can differ slightly.
MEDIAN_TOLERANCE_MS = 1e-3


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
    trial_latencies_ms: list[Annotated[float, Field(ge=0)]] = Field(default_factory=list)
    correct: bool | None = None
    started_unix_ms: float | None = Field(default=None, ge=0)
    finished_unix_ms: float | None = Field(default=None, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_availability_fields(self) -> "Measurement":
        if self.available:
            if (
                self.latency_ms is None
                or self.correct is None
                or self.started_unix_ms is None
                or self.finished_unix_ms is None
            ):
                raise ValueError(
                    "available measurements require latency_ms, correct, and a "
                    "timing window"
                )
            if not self.trial_latencies_ms:
                raise ValueError("available measurements require trial latencies")
            if abs(median(self.trial_latencies_ms) - self.latency_ms) > MEDIAN_TOLERANCE_MS:
                raise ValueError("latency_ms must be the median of trial_latencies_ms")
            if self.finished_unix_ms < self.started_unix_ms:
                raise ValueError("finished_unix_ms cannot precede started_unix_ms")
            if self.error is not None:
                raise ValueError("available measurements cannot contain an error")
        else:
            if (
                self.latency_ms is not None
                or self.correct is not None
                or self.trial_latencies_ms
                or self.started_unix_ms is not None
                or self.finished_unix_ms is not None
            ):
                raise ValueError(
                    "unavailable measurements cannot contain timings or correctness"
                )
            if not self.error:
                raise ValueError("unavailable measurements require an error")
        return self


class GpuDevice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    name: str
    memory_total_mib: float = Field(ge=0)


class GpuPhaseSummary(BaseModel):
    """NVML samples for one GPU during one benchmark phase.

    Statistics are None when the phase had too few samples to summarize.
    """

    model_config = ConfigDict(extra="forbid")

    phase: TelemetryPhase
    gpu_index: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    utilization_mean_percent: float | None = None
    utilization_max_percent: float | None = None
    memory_used_max_mib: float | None = None
    temperature_max_c: float | None = None
    power_mean_w: float | None = None
    power_max_w: float | None = None


class GpuTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    error: str | None = None
    sample_interval_ms: float = Field(gt=0)
    min_phase_samples: int = Field(ge=1)
    devices: list[GpuDevice] = Field(default_factory=list)
    phases: list[GpuPhaseSummary] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix_size: MatrixSize
    repeats: int = Field(ge=1)
    data_type: Literal["float32"]
    transfer_included: bool
    cuda_context_warmup_excluded: bool
    gpu_name: str
    gpu_count: int = Field(ge=0)
    # Team size of a real OpenMP parallel region, not omp_get_max_threads().
    openmp_threads: int = Field(ge=1)
    measurements: list[Measurement] = Field(
        min_length=len(IMPLEMENTATIONS), max_length=len(IMPLEMENTATIONS)
    )
    # Filled in by the Python runner; the native benchmark never reports it.
    telemetry: GpuTelemetry | None = None

    @model_validator(mode="after")
    def require_each_implementation_once(self) -> "BenchmarkResult":
        implementations = [item.implementation for item in self.measurements]
        if sorted(implementations) != sorted(IMPLEMENTATIONS):
            raise ValueError(
                "results must contain CPU, OpenMP, CUDA, and multi-GPU CUDA "
                "exactly once"
            )
        for item in self.measurements:
            if item.available and len(item.trial_latencies_ms) != self.repeats:
                raise ValueError(
                    f"{item.implementation} must report exactly {self.repeats} trials"
                )
        if self.measurement("cuda").available and self.gpu_count < 1:
            raise ValueError("a CUDA measurement requires at least one GPU")
        if self.measurement("cuda_multi_gpu").available and self.gpu_count < 2:
            raise ValueError("a multi-GPU measurement requires at least two GPUs")
        return self

    def measurement(self, implementation: ImplementationName) -> Measurement:
        return next(
            item for item in self.measurements if item.implementation == implementation
        )


class Analysis(BaseModel):
    fastest_implementation: ImplementationName | None
    speedups: dict[str, float]
    multi_gpu_scaling_efficiency: float | None
    correctness_passed: bool
    complete_comparison: bool
    multi_gpu_compared: bool
    recommendation_allowed: bool
    notes: list[str]


class AdviceResponse(BaseModel):
    benchmark: BenchmarkResult
    analysis: Analysis
    explanation: str
    report: str
