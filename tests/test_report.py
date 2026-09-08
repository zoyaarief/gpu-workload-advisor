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
    assert "single run does not establish performance" in report
    assert "./build/benchmark_runner 1024" in report
