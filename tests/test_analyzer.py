from __future__ import annotations

from app.analyzer import analyze
from app.schemas import BenchmarkResult


def test_analyzer_calculates_speedups_from_measurements(sample_result) -> None:
    analysis = analyze(sample_result)

    assert analysis.fastest_implementation == "cuda"
    assert analysis.speedups == {
        "openmp_vs_cpu": 4.0,
        "cuda_vs_cpu": 12.0,
        "cuda_vs_openmp": 3.0,
    }
    assert analysis.correctness_passed is True
    assert analysis.recommendation_allowed is True


def test_analyzer_blocks_recommendation_when_cuda_is_unavailable(
    sample_result,
) -> None:
    data = sample_result.model_dump()
    data["measurements"][2] = {
        "implementation": "cuda",
        "available": False,
        "latency_ms": None,
        "correct": None,
        "error": "No CUDA-capable GPU was detected",
    }

    analysis = analyze(BenchmarkResult.model_validate(data))

    assert analysis.complete_comparison is False
    assert analysis.correctness_passed is False
    assert analysis.recommendation_allowed is False
    assert "cuda_vs_cpu" not in analysis.speedups


def test_analyzer_blocks_speedup_for_incorrect_result(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][2]["correct"] = False

    analysis = analyze(BenchmarkResult.model_validate(data))

    assert analysis.recommendation_allowed is False
    assert analysis.fastest_implementation == "openmp"
    assert "cuda_vs_cpu" not in analysis.speedups
