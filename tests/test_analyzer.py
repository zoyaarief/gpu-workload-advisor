from __future__ import annotations

from app.analyzer import analyze
from app.schemas import BenchmarkResult, GpuTelemetry

from tests.conftest import unavailable


def test_analyzer_calculates_speedups_from_measurements(sample_result) -> None:
    analysis = analyze(sample_result)

    assert analysis.fastest_implementation == "cuda_multi_gpu"
    assert analysis.speedups == {
        "openmp_vs_cpu": 4.0,
        "cuda_vs_cpu": 12.0,
        "cuda_vs_openmp": 3.0,
        "cuda_multi_gpu_vs_cpu": 20.0,
        "cuda_multi_gpu_vs_cuda": 1.667,
    }
    assert analysis.multi_gpu_scaling_efficiency == 0.833
    assert analysis.multi_gpu_compared is True
    assert analysis.correctness_passed is True
    assert analysis.recommendation_allowed is True
    assert "latencies are medians of 3 trials" in analysis.notes


def test_analyzer_blocks_recommendation_when_cuda_is_unavailable(
    sample_result,
) -> None:
    data = sample_result.model_dump()
    data["measurements"][2] = unavailable("cuda", "No CUDA-capable GPU was detected")

    analysis = analyze(BenchmarkResult.model_validate(data))

    assert analysis.complete_comparison is False
    assert analysis.correctness_passed is False
    assert analysis.recommendation_allowed is False
    assert "cuda_vs_cpu" not in analysis.speedups
    assert analysis.multi_gpu_scaling_efficiency is None


def test_analyzer_blocks_speedup_for_incorrect_result(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][2]["correct"] = False

    analysis = analyze(BenchmarkResult.model_validate(data))

    assert analysis.recommendation_allowed is False
    assert analysis.fastest_implementation == "cuda_multi_gpu"
    assert "cuda_vs_cpu" not in analysis.speedups
    assert "cuda_multi_gpu_vs_cuda" not in analysis.speedups


def test_single_gpu_machine_still_gets_a_recommendation(sample_result) -> None:
    data = sample_result.model_dump()
    data["gpu_count"] = 1
    data["measurements"][3] = unavailable(
        "cuda_multi_gpu",
        "multi-GPU run requires at least two CUDA devices; found 1",
    )

    analysis = analyze(BenchmarkResult.model_validate(data))

    assert analysis.complete_comparison is True
    assert analysis.recommendation_allowed is True
    assert analysis.fastest_implementation == "cuda"
    assert analysis.multi_gpu_compared is False
    assert analysis.multi_gpu_scaling_efficiency is None
    assert any("found 1" in note for note in analysis.notes)


def test_incorrect_multi_gpu_result_blocks_recommendation(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][3]["correct"] = False

    analysis = analyze(BenchmarkResult.model_validate(data))

    assert analysis.recommendation_allowed is False
    assert analysis.fastest_implementation == "cuda"
    assert "cuda_multi_gpu failed the correctness check" in analysis.notes


def test_single_threaded_openmp_is_flagged(sample_result) -> None:
    single_thread = sample_result.model_copy(update={"openmp_threads": 1})

    notes = analyze(single_thread).notes

    assert any("OpenMP ran with one thread" in note for note in notes)
    assert not any("OpenMP ran with one thread" in note for note in analyze(sample_result).notes)


def test_analyzer_explains_missing_telemetry(sample_result) -> None:
    missing = sample_result.model_copy(
        update={
            "telemetry": GpuTelemetry(
                available=False,
                error="NVML unavailable: NVML Shared Library Not Found",
                sample_interval_ms=50,
                min_phase_samples=2,
            )
        }
    )

    assert (
        "GPU telemetry unavailable: NVML unavailable: NVML Shared Library Not Found"
        in analyze(missing).notes
    )
    assert "GPU telemetry was not collected" in analyze(sample_result).notes


def test_analyzer_flags_under_sampled_phases(sample_result, sample_telemetry) -> None:
    with_telemetry = sample_result.model_copy(update={"telemetry": sample_telemetry})

    notes = analyze(with_telemetry).notes

    assert (
        "cuda_multi_gpu telemetry on GPU 1 has 1 samples, too few to summarize"
        in notes
    )
    assert not any("cuda telemetry" in note for note in notes)
