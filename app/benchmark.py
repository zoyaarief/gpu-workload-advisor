from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.schemas import BenchmarkResult


class BenchmarkError(Exception):
    """Base class for benchmark failures safe to map to an API response."""


class BenchmarkInputError(BenchmarkError):
    pass


class BenchmarkUnavailableError(BenchmarkError):
    pass


class BenchmarkExecutionError(BenchmarkError):
    pass


class BenchmarkRunner:
    """Runs one fixed executable with one validated integer argument."""

    def __init__(self, settings: Settings) -> None:
        self._executable = settings.benchmark_executable
        self._max_matrix_size = settings.max_matrix_size
        self._timeout_seconds = settings.benchmark_timeout_seconds

    @property
    def executable(self) -> Path:
        return self._executable

    def run(self, matrix_size: int) -> BenchmarkResult:
        self._validate_size(matrix_size)
        self._validate_executable()

        command = [str(self._executable), str(matrix_size)]
        try:
            completed = subprocess.run(
                command,
                cwd=self._executable.parent,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise BenchmarkExecutionError(
                f"benchmark exceeded the {self._timeout_seconds}-second timeout"
            ) from error
        except OSError as error:
            raise BenchmarkExecutionError("benchmark process could not start") from error

        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown native benchmark error"
            raise BenchmarkExecutionError(f"benchmark failed: {detail}")

        try:
            result = BenchmarkResult.model_validate_json(completed.stdout)
        except ValidationError as error:
            raise BenchmarkExecutionError(
                "benchmark returned malformed or incomplete JSON"
            ) from error

        if result.matrix_size != matrix_size:
            raise BenchmarkExecutionError(
                "benchmark returned results for a different matrix size"
            )
        return result

    def _validate_size(self, matrix_size: int) -> None:
        if isinstance(matrix_size, bool) or not isinstance(matrix_size, int):
            raise BenchmarkInputError("matrix_size must be an integer")
        if matrix_size < 1 or matrix_size > self._max_matrix_size:
            raise BenchmarkInputError(
                f"matrix_size must be between 1 and {self._max_matrix_size}"
            )

    def _validate_executable(self) -> None:
        if not self._executable.is_file() or not os.access(self._executable, os.X_OK):
            raise BenchmarkUnavailableError(
                "benchmark executable is missing; build it with CMake first"
            )
