from __future__ import annotations

from app.schemas import Analysis, BenchmarkResult, Measurement


def analyze(result: BenchmarkResult) -> Analysis:
    by_name = {item.implementation: item for item in result.measurements}
    available = [item for item in result.measurements if item.available]
    complete = len(available) == 3
    correctness_passed = complete and all(item.correct is True for item in available)

    fastest: str | None = None
    valid_timings = [
        item
        for item in available
        if item.correct is True and item.latency_ms is not None
    ]
    if valid_timings:
        fastest = min(valid_timings, key=lambda item: item.latency_ms or 0).implementation

    speedups: dict[str, float] = {}
    _add_speedup(speedups, "openmp_vs_cpu", by_name["cpu"], by_name["openmp"])
    _add_speedup(speedups, "cuda_vs_cpu", by_name["cpu"], by_name["cuda"])
    _add_speedup(speedups, "cuda_vs_openmp", by_name["openmp"], by_name["cuda"])

    notes: list[str] = []
    for measurement in result.measurements:
        if not measurement.available:
            notes.append(
                f"{measurement.implementation} unavailable: {measurement.error}"
            )
        elif measurement.correct is not True:
            notes.append(
                f"{measurement.implementation} failed the correctness check"
            )

    if result.transfer_included:
        notes.append("CUDA latency includes allocation and host/device transfers")
    else:
        notes.append("CUDA latency excludes host/device transfer overhead")
    if result.cuda_context_warmup_excluded:
        notes.append("one-time CUDA context initialization is excluded")

    return Analysis(
        fastest_implementation=fastest,  # type: ignore[arg-type]
        speedups=speedups,
        correctness_passed=correctness_passed,
        complete_comparison=complete,
        recommendation_allowed=complete and correctness_passed,
        notes=notes,
    )


def _add_speedup(
    destination: dict[str, float],
    name: str,
    baseline: Measurement,
    candidate: Measurement,
) -> None:
    if not (
        baseline.available
        and candidate.available
        and baseline.correct is True
        and candidate.correct is True
        and baseline.latency_ms is not None
        and candidate.latency_ms is not None
        and candidate.latency_ms > 0
    ):
        return
    destination[name] = round(baseline.latency_ms / candidate.latency_ms, 3)
