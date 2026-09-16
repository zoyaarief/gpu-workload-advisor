from __future__ import annotations

import pytest

from app.schemas import BenchmarkResult, GpuTelemetry


START_MS = 1_700_000_000_000.0


def measurement(
    implementation: str,
    trials: list[float],
    window: tuple[float, float],
) -> dict:
    return {
        "implementation": implementation,
        "available": True,
        "latency_ms": sorted(trials)[len(trials) // 2],
        "trial_latencies_ms": trials,
        "correct": True,
        "started_unix_ms": START_MS + window[0],
        "finished_unix_ms": START_MS + window[1],
        "error": None,
    }


def unavailable(implementation: str, error: str) -> dict:
    return {
        "implementation": implementation,
        "available": False,
        "latency_ms": None,
        "trial_latencies_ms": [],
        "correct": None,
        "started_unix_ms": None,
        "finished_unix_ms": None,
        "error": error,
    }


@pytest.fixture
def sample_result() -> BenchmarkResult:
    return BenchmarkResult.model_validate(
        {
            "matrix_size": 1024,
            "repeats": 3,
            "data_type": "float32",
            "transfer_included": True,
            "cuda_context_warmup_excluded": True,
            "gpu_name": "Test GPU",
            "gpu_count": 2,
            "openmp_threads": 8,
            "measurements": [
                measurement("cpu", [121.0, 120.0, 119.0], (0, 400)),
                measurement("openmp", [30.0, 31.0, 29.0], (400, 500)),
                measurement("cuda", [10.0, 10.5, 9.5], (500, 540)),
                measurement("cuda_multi_gpu", [6.0, 6.2, 5.8], (540, 560)),
            ],
        }
    )


@pytest.fixture
def sample_telemetry() -> GpuTelemetry:
    def phase(name: str, gpu_index: int, utilization: float) -> dict:
        return {
            "phase": name,
            "gpu_index": gpu_index,
            "sample_count": 4,
            "utilization_mean_percent": utilization,
            "utilization_max_percent": utilization,
            "memory_used_max_mib": 512.0,
            "temperature_max_c": 60.0,
            "power_mean_w": 70.5,
            "power_max_w": 75.0,
        }

    return GpuTelemetry.model_validate(
        {
            "available": True,
            "sample_interval_ms": 50,
            "min_phase_samples": 2,
            "devices": [
                {"index": 0, "name": "Tesla T4", "memory_total_mib": 15360},
                {"index": 1, "name": "Tesla T4", "memory_total_mib": 15360},
            ],
            "phases": [
                phase("idle", 0, 0),
                phase("idle", 1, 0),
                phase("cuda", 0, 97),
                phase("cuda", 1, 0),
                phase("cuda_multi_gpu", 0, 88),
                {"phase": "cuda_multi_gpu", "gpu_index": 1, "sample_count": 1},
            ],
        }
    )
