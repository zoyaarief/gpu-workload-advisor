from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from statistics import fmean
from types import TracebackType
from typing import Any, Protocol

from app.schemas import (
    GpuDevice,
    GpuPhaseSummary,
    GpuTelemetry,
    Measurement,
    TelemetryPhase,
)


MIN_PHASE_SAMPLES = 2
BASELINE_SAMPLES = 3
GPU_PHASES: tuple[TelemetryPhase, ...] = ("cuda", "cuda_multi_gpu")
MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuSample:
    timestamp_unix_ms: float
    gpu_index: int
    utilization_percent: float
    memory_used_mib: float
    temperature_c: float
    power_w: float | None


class TelemetrySampler(Protocol):
    def __enter__(self) -> "TelemetrySampler": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def telemetry(self, measurements: list[Measurement]) -> GpuTelemetry: ...


def load_nvml() -> Any:
    import pynvml

    return pynvml


class NvmlTelemetrySampler:
    """Samples every GPU through NVML on a background thread.

    Telemetry is best effort: when NVML is missing or fails, the benchmark still
    runs and the result says why telemetry is unavailable.
    """

    def __init__(self, interval_ms: float, nvml: Any | None = None) -> None:
        if interval_ms <= 0:
            raise ValueError("telemetry interval must be positive")
        self._interval_ms = interval_ms
        self._nvml = nvml
        self._initialized = False
        self._handles: list[Any] = []
        self._devices: list[GpuDevice] = []
        self._baseline: list[GpuSample] = []
        self._samples: list[GpuSample] = []
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "NvmlTelemetrySampler":
        try:
            nvml = self._nvml if self._nvml is not None else load_nvml()
            nvml.nvmlInit()
        except Exception as error:
            self._error = f"NVML unavailable: {error}"
            return self
        self._nvml = nvml
        self._initialized = True

        try:
            self._handles = [
                nvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(nvml.nvmlDeviceGetCount())
            ]
            self._devices = [
                GpuDevice(
                    index=index,
                    name=_text(nvml.nvmlDeviceGetName(handle)),
                    memory_total_mib=nvml.nvmlDeviceGetMemoryInfo(handle).total / MIB,
                )
                for index, handle in enumerate(self._handles)
            ]
            for _ in range(BASELINE_SAMPLES):
                self._baseline.extend(self._read_all())
                time.sleep(self._interval_ms / 1000)
        except Exception as error:
            self._error = f"NVML query failed: {error}"
            return self

        self._thread = threading.Thread(
            target=self._sample_until_stopped, name="gpu-telemetry", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def telemetry(self, measurements: list[Measurement]) -> GpuTelemetry:
        if self._error is not None:
            return GpuTelemetry(
                available=False,
                error=self._error,
                sample_interval_ms=self._interval_ms,
                min_phase_samples=MIN_PHASE_SAMPLES,
            )
        return GpuTelemetry(
            available=True,
            sample_interval_ms=self._interval_ms,
            min_phase_samples=MIN_PHASE_SAMPLES,
            devices=self._devices,
            phases=summarize_phases(
                self._baseline,
                self._samples,
                measurements,
                [device.index for device in self._devices],
            ),
        )

    def _sample_until_stopped(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.extend(self._read_all())
            except Exception as error:
                self._error = f"NVML sampling failed: {error}"
                return
            self._stop.wait(self._interval_ms / 1000)

    def _read_all(self) -> list[GpuSample]:
        nvml = self._nvml
        timestamp = time.time() * 1000
        samples = []
        for index, handle in enumerate(self._handles):
            try:
                power_w: float | None = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except nvml.NVMLError:
                power_w = None
            samples.append(
                GpuSample(
                    timestamp_unix_ms=timestamp,
                    gpu_index=index,
                    utilization_percent=nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    memory_used_mib=nvml.nvmlDeviceGetMemoryInfo(handle).used / MIB,
                    temperature_c=nvml.nvmlDeviceGetTemperature(
                        handle, nvml.NVML_TEMPERATURE_GPU
                    ),
                    power_w=power_w,
                )
            )
        return samples


def summarize_phases(
    baseline: list[GpuSample],
    samples: list[GpuSample],
    measurements: list[Measurement],
    gpu_indices: list[int],
) -> list[GpuPhaseSummary]:
    """Attributes samples to the native timing window of each GPU phase."""
    phases: list[tuple[TelemetryPhase, list[GpuSample]]] = [("idle", baseline)]
    for measurement in measurements:
        if measurement.implementation not in GPU_PHASES or not measurement.available:
            continue
        start = measurement.started_unix_ms
        end = measurement.finished_unix_ms
        phases.append(
            (
                measurement.implementation,  # type: ignore[arg-type]
                [
                    sample
                    for sample in samples
                    if start <= sample.timestamp_unix_ms <= end  # type: ignore[operator]
                ],
            )
        )

    return [
        _summarize(
            phase,
            gpu_index,
            [sample for sample in phase_samples if sample.gpu_index == gpu_index],
        )
        for phase, phase_samples in phases
        for gpu_index in gpu_indices
    ]


def _summarize(
    phase: TelemetryPhase, gpu_index: int, samples: list[GpuSample]
) -> GpuPhaseSummary:
    if len(samples) < MIN_PHASE_SAMPLES:
        return GpuPhaseSummary(
            phase=phase, gpu_index=gpu_index, sample_count=len(samples)
        )

    utilization = [sample.utilization_percent for sample in samples]
    power = [sample.power_w for sample in samples if sample.power_w is not None]
    return GpuPhaseSummary(
        phase=phase,
        gpu_index=gpu_index,
        sample_count=len(samples),
        utilization_mean_percent=round(fmean(utilization), 1),
        utilization_max_percent=max(utilization),
        memory_used_max_mib=round(max(sample.memory_used_mib for sample in samples), 1),
        temperature_max_c=max(sample.temperature_c for sample in samples),
        power_mean_w=round(fmean(power), 1) if power else None,
        power_max_w=round(max(power), 1) if power else None,
    )


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value
