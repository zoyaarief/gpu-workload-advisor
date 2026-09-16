from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BenchmarkRequest, BenchmarkResult

from tests.conftest import unavailable


@pytest.mark.parametrize("value", [0, -1, 2.5, "1024", True])
def test_matrix_size_is_a_strict_positive_integer(value) -> None:
    with pytest.raises(ValidationError):
        BenchmarkRequest(matrix_size=value)


def test_result_requires_each_implementation_exactly_once(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][3]["implementation"] = "cuda"

    with pytest.raises(ValidationError, match="exactly once"):
        BenchmarkResult.model_validate(data)


def test_latency_must_be_the_median_of_trials(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][2]["latency_ms"] = 9.5

    with pytest.raises(ValidationError, match="median"):
        BenchmarkResult.model_validate(data)


def test_result_requires_the_reported_trial_count(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][0]["trial_latencies_ms"] = [120.0]
    data["measurements"][0]["latency_ms"] = 120.0

    with pytest.raises(ValidationError, match="exactly 3 trials"):
        BenchmarkResult.model_validate(data)


def test_timing_window_cannot_run_backwards(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][2]["finished_unix_ms"] = (
        data["measurements"][2]["started_unix_ms"] - 1
    )

    with pytest.raises(ValidationError, match="cannot precede"):
        BenchmarkResult.model_validate(data)


def test_unavailable_measurement_cannot_carry_trials(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][3] = unavailable("cuda_multi_gpu", "one GPU")
    data["measurements"][3]["trial_latencies_ms"] = [1.0]

    with pytest.raises(ValidationError, match="cannot contain timings"):
        BenchmarkResult.model_validate(data)


def test_multi_gpu_measurement_requires_two_gpus(sample_result) -> None:
    data = sample_result.model_dump()
    data["gpu_count"] = 1

    with pytest.raises(ValidationError, match="at least two GPUs"):
        BenchmarkResult.model_validate(data)


def test_native_output_does_not_require_telemetry(sample_result) -> None:
    data = sample_result.model_dump()
    del data["telemetry"]

    assert BenchmarkResult.model_validate(data).telemetry is None
