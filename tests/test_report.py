from __future__ import annotations

from app.analyzer import analyze
from app.report import generate_report


def test_report_discloses_measurement_policy(sample_result) -> None:
    report = generate_report(
        sample_result,
        analyze(sample_result),
        "CUDA was fastest in this measured run.",
    )

    assert "CUDA allocation and host/device transfers: included" in report
    assert "One-time CUDA context initialization: excluded" in report
    assert "Each latency is the median of 3 trials." in report
    assert "trials on one machine do not establish performance" in report
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/benchmark_runner 1024 3" in report


def test_report_shows_trial_spread_and_multi_gpu_scaling(sample_result) -> None:
    report = generate_report(sample_result, analyze(sample_result), "Explained.")

    assert "| CPU | 120.000 | 119.000–121.000 | pass | available |" in report
    assert "| CUDA (multi-GPU) | 6.000 | 5.800–6.200 | pass | available |" in report
    assert "`cuda_multi_gpu_vs_cuda`: 1.667×" in report
    assert "Multi-GPU scaling efficiency: 0.833" in report
    assert "÷ 2 GPUs" in report
    assert "fastest correct implementation in this run was **CUDA (multi-GPU)**" in report
    assert "GPU telemetry was not collected." in report


def test_report_renders_telemetry_table(sample_result, sample_telemetry) -> None:
    measured = sample_result.model_copy(update={"telemetry": sample_telemetry})

    report = generate_report(measured, analyze(measured), "Explained.")

    assert "Sampled with NVML every 50 ms." in report
    assert "- GPU 1: Tesla T4, 15360 MiB" in report
    assert "| CUDA (1 GPU) | 0 | 4 | 97 | 97 | 512 | 60 | 70.5 | 75 |" in report
    assert (
        "| CUDA (multi-GPU) | 1 | 1 | N/A | N/A | N/A | N/A | N/A | N/A |" in report
    )


def test_report_flags_single_threaded_openmp(sample_result) -> None:
    single_thread = sample_result.model_copy(update={"openmp_threads": 1})

    report = generate_report(single_thread, analyze(single_thread), "Explained.")

    assert "OpenMP threads used: 1 (OpenMP did not parallelize" in report
