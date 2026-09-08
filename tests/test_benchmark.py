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


def make_runner(tmp_path, maximum: int = 2048) -> BenchmarkRunner:
    executable = tmp_path / "benchmark_runner"
    executable.write_text("placeholder")
    executable.chmod(0o755)
    return BenchmarkRunner(
        Settings(
            benchmark_executable=executable,
            max_matrix_size=maximum,
            benchmark_timeout_seconds=5,
            openai_model="test-model",
        )
    )


@pytest.mark.parametrize("matrix_size", [0, -1, 2049, True, 12.5, "32"])
def test_runner_rejects_invalid_sizes(tmp_path, matrix_size) -> None:
    runner = make_runner(tmp_path)
    with pytest.raises(BenchmarkInputError):
        runner.run(matrix_size)


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
    assert captured["command"] == [str(runner.executable), "1024"]
    assert captured["kwargs"]["shell"] is False


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


def test_runner_reports_missing_executable(tmp_path) -> None:
    runner = BenchmarkRunner(
        Settings(
            benchmark_executable=tmp_path / "missing",
            max_matrix_size=2048,
            benchmark_timeout_seconds=5,
            openai_model="test-model",
        )
    )

    with pytest.raises(BenchmarkUnavailableError):
        runner.run(32)
