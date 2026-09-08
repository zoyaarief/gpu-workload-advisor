from __future__ import annotations

import pytest

from app.schemas import BenchmarkResult


@pytest.fixture
def sample_result() -> BenchmarkResult:
    return BenchmarkResult.model_validate(
        {
            "matrix_size": 1024,
            "data_type": "float32",
            "transfer_included": True,
            "cuda_context_warmup_excluded": True,
            "gpu_name": "Test GPU",
            "openmp_threads": 8,
            "measurements": [
                {
                    "implementation": "cpu",
                    "available": True,
                    "latency_ms": 120.0,
                    "correct": True,
                    "error": None,
                },
                {
                    "implementation": "openmp",
                    "available": True,
                    "latency_ms": 30.0,
                    "correct": True,
                    "error": None,
                },
                {
                    "implementation": "cuda",
                    "available": True,
                    "latency_ms": 10.0,
                    "correct": True,
                    "error": None,
                },
            ],
        }
    )
