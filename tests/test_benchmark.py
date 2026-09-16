from __future__ import annotations

import json
import subprocess

import pytest

from app.benchmark import (
    BenchmarkExecutionError,
    BenchmarkInputError,
    BenchmarkRunner,
    BenchmarkUnavailableError,
)
from app.config import Settings
from app.schemas import GpuTelemetry


class FakeSampler:
    def __init__(self, telemetry: GpuTelemetry) -> None:
        self.result = telemetry
        self.active = False
        self.received = None

    def __enter__(self) -> "FakeSampler":
        self.active = True
        return self

    def __exit__(self, *exc_info) -> None:
        self.active = False

    def telemetry(self, measurements):
        self.received = measurements
        return self.result


def unavailable_telemetry() -> GpuTelemetry:
    return GpuTelemetry(
        available=False,
        error="NVML unavailable: test",
        sample_interval_ms=50,
        min_phase_samples=2,
    )


def make_runner(
    tmp_path,
    maximum: int = 2048,
    repeats: int = 3,
    sampler: FakeSampler | None = None,
) -> BenchmarkRunner:
    executable = tmp_path / "benchmark_runner"
    executable.write_text("placeholder")
    executable.chmod(0o755)
    sampler = sampler or FakeSampler(unavailable_telemetry())
    return BenchmarkRunner(
        Settings(
            benchmark_executable=executable,
            max_matrix_size=maximum,
            benchmark_timeout_seconds=5,
            benchmark_repeats=repeats,
            groq_model="test-model",
        ),
        sampler_factory=lambda: sampler,
    )


@pytest.mark.parametrize("matrix_size", [0, -1, 2049, True, 12.5, "32"])
def test_runner_rejects_invalid_sizes(tmp_path, matrix_size) -> None:
    runner = make_runner(tmp_path)
    with pytest.raises(BenchmarkInputError):
        runner.run(matrix_size)


@pytest.mark.parametrize("repeats", [0, 51])
def test_runner_rejects_invalid_trial_counts(tmp_path, repeats) -> None:
    with pytest.raises(ValueError, match="benchmark_repeats"):
        make_runner(tmp_path, repeats=repeats)


def test_runner_uses_fixed_executable_without_a_shell(
    tmp_path, monkeypatch, sample_result
) -> None:
    runner = make_runner(tmp_path)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, 0, stdout=sample_result.model_dump_json(), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.run(1024)

    assert result.matrix_size == 1024
    assert captured["command"] == [str(runner.executable), "1024", "3"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_runner_samples_telemetry_while_the_benchmark_runs(
    tmp_path, monkeypatch, sample_result, sample_telemetry
) -> None:
    sampler = FakeSampler(sample_telemetry)
    runner = make_runner(tmp_path, sampler=sampler)

    def fake_run(command, **kwargs):
        assert sampler.active, "the benchmark must run inside the sampling window"
        return subprocess.CompletedProcess(
            command, 0, stdout=sample_result.model_dump_json(), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.run(1024)

    assert sampler.active is False
    assert result.telemetry == sample_telemetry
    assert sampler.received == result.measurements


def test_runner_ignores_telemetry_claimed_by_native_output(
    tmp_path, monkeypatch, sample_result, sample_telemetry
) -> None:
    runner = make_runner(tmp_path)
    claimed = sample_result.model_copy(update={"telemetry": sample_telemetry})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=claimed.model_dump_json(), stderr=""
        ),
    )

    assert runner.run(1024).telemetry == unavailable_telemetry()


def test_runner_rejects_a_different_trial_count(
    tmp_path, monkeypatch, sample_result
) -> None:
    runner = make_runner(tmp_path, repeats=5)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=sample_result.model_dump_json(), stderr=""
        ),
    )

    with pytest.raises(BenchmarkExecutionError, match="number of trials"):
        runner.run(1024)


def test_runner_rejects_malformed_native_output(tmp_path, monkeypatch) -> None:
    runner = make_runner(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps({"matrix_size": 1024}), stderr=""
        ),
    )

    with pytest.raises(BenchmarkExecutionError, match="malformed"):
        runner.run(1024)


def test_runner_reports_native_failures(tmp_path, monkeypatch) -> None:
    runner = make_runner(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="", stderr="Benchmark failed: repeats must be an integer"
        ),
    )

    with pytest.raises(BenchmarkExecutionError, match="repeats must be an integer"):
        runner.run(1024)


def test_runner_reports_missing_executable(tmp_path) -> None:
    runner = BenchmarkRunner(
        Settings(
            benchmark_executable=tmp_path / "missing",
            max_matrix_size=2048,
            benchmark_timeout_seconds=5,
            groq_model="test-model",
        )
    )

    with pytest.raises(BenchmarkUnavailableError):
        runner.run(32)
