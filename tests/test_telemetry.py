from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.schemas import Measurement
from app.telemetry import (
    MIN_PHASE_SAMPLES,
    GpuSample,
    NvmlTelemetrySampler,
    summarize_phases,
)

from tests.conftest import unavailable


class FakeNvmlError(Exception):
    pass


class FakeNvml:
    """Two GPUs; GPU 1 does not support power readings."""

    NVMLError = FakeNvmlError
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, fail_init: bool = False, fail_after_reads: int | None = None):
        self.fail_init = fail_init
        self.fail_after_reads = fail_after_reads
        self.utilization_reads = 0
        self.shutdown_calls = 0

    def nvmlInit(self) -> None:
        if self.fail_init:
            raise FakeNvmlError("NVML Shared Library Not Found")

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetName(self, handle: int) -> bytes:
        return b"Tesla T4"

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(total=16 * 1024**3, used=(256 + handle) * 1024**2)

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> SimpleNamespace:
        self.utilization_reads += 1
        if self.fail_after_reads is not None and self.utilization_reads > self.fail_after_reads:
            raise FakeNvmlError("GPU is lost")
        return SimpleNamespace(gpu=90 if handle == 0 else 40)

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        assert sensor == self.NVML_TEMPERATURE_GPU
        return 55 + handle

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        if handle == 1:
            raise FakeNvmlError("Not Supported")
        return 70_000


def window(implementation: str, start: float, end: float) -> Measurement:
    return Measurement(
        implementation=implementation,
        available=True,
        latency_ms=1.0,
        trial_latencies_ms=[1.0],
        correct=True,
        started_unix_ms=start,
        finished_unix_ms=end,
    )


def sample(timestamp: float, gpu_index: int, utilization: float) -> GpuSample:
    return GpuSample(
        timestamp_unix_ms=timestamp,
        gpu_index=gpu_index,
        utilization_percent=utilization,
        memory_used_mib=100 + utilization,
        temperature_c=50 + gpu_index,
        power_w=None if gpu_index == 1 else utilization,
    )


def test_sampler_collects_samples_during_the_benchmark() -> None:
    nvml = FakeNvml()
    sampler = NvmlTelemetrySampler(interval_ms=5, nvml=nvml)

    with sampler:
        started = time.time() * 1000
        time.sleep(0.1)
        finished = time.time() * 1000

    telemetry = sampler.telemetry(
        [
            window("cpu", started, finished),
            window("cuda_multi_gpu", started, finished),
            Measurement.model_validate(unavailable("cuda", "no GPU")),
        ]
    )

    assert telemetry.available is True
    assert [device.name for device in telemetry.devices] == ["Tesla T4", "Tesla T4"]
    assert telemetry.devices[0].memory_total_mib == 16 * 1024
    assert [(phase.phase, phase.gpu_index) for phase in telemetry.phases] == [
        ("idle", 0),
        ("idle", 1),
        ("cuda_multi_gpu", 0),
        ("cuda_multi_gpu", 1),
    ]
    idle, _, busy_gpu0, busy_gpu1 = telemetry.phases
    assert idle.sample_count == 3
    assert busy_gpu0.sample_count >= MIN_PHASE_SAMPLES
    assert busy_gpu0.utilization_max_percent == 90
    assert busy_gpu0.memory_used_max_mib == 256
    assert busy_gpu0.power_max_w == 70
    assert busy_gpu1.temperature_max_c == 56
    assert busy_gpu1.power_mean_w is None
    assert nvml.shutdown_calls == 1


def test_sampler_reports_missing_nvml_without_failing() -> None:
    nvml = FakeNvml(fail_init=True)
    sampler = NvmlTelemetrySampler(interval_ms=5, nvml=nvml)

    with sampler:
        pass

    telemetry = sampler.telemetry([])
    assert telemetry.available is False
    assert telemetry.error == "NVML unavailable: NVML Shared Library Not Found"
    assert telemetry.phases == []
    assert nvml.shutdown_calls == 0


def test_sampler_reports_a_failure_during_sampling() -> None:
    # Baseline reads 2 GPUs x 3 sweeps; the first background read then fails.
    nvml = FakeNvml(fail_after_reads=6)
    sampler = NvmlTelemetrySampler(interval_ms=5, nvml=nvml)

    with sampler:
        time.sleep(0.05)

    telemetry = sampler.telemetry([])
    assert telemetry.available is False
    assert telemetry.error == "NVML sampling failed: GPU is lost"
    assert nvml.shutdown_calls == 1


def test_sampler_requires_a_positive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        NvmlTelemetrySampler(interval_ms=0, nvml=FakeNvml())


def test_phases_only_include_samples_inside_their_window() -> None:
    baseline = [sample(0, 0, 0), sample(0, 1, 0), sample(5, 0, 2), sample(5, 1, 0)]
    samples = [
        sample(99, 0, 50),  # before the CUDA window
        sample(100, 0, 60),
        sample(150, 0, 80),
        sample(150, 1, 5),
        sample(200, 0, 100),
        sample(201, 0, 0),  # after the CUDA window
        sample(300, 0, 70),
        sample(300, 1, 75),
        sample(320, 1, 85),
    ]

    phases = summarize_phases(
        baseline,
        samples,
        [
            window("cuda", 100, 200),
            window("cuda_multi_gpu", 300, 320),
        ],
        [0, 1],
    )
    by_key = {(phase.phase, phase.gpu_index): phase for phase in phases}

    idle = by_key[("idle", 0)]
    assert (idle.sample_count, idle.utilization_mean_percent) == (2, 1.0)

    single = by_key[("cuda", 0)]
    assert single.sample_count == 3
    assert single.utilization_mean_percent == 80.0
    assert single.utilization_max_percent == 100
    assert single.memory_used_max_mib == 200
    assert single.power_mean_w == 80.0

    # One sample is too few to summarize.
    quiet = by_key[("cuda", 1)]
    assert quiet.sample_count == 1
    assert quiet.utilization_mean_percent is None

    multi = by_key[("cuda_multi_gpu", 1)]
    assert multi.sample_count == 2
    assert multi.utilization_max_percent == 85
    assert multi.power_max_w is None
