from __future__ import annotations

from app.schemas import Analysis, BenchmarkResult, Measurement


CORE_IMPLEMENTATIONS = ("cpu", "openmp", "cuda")


def analyze(result: BenchmarkResult) -> Analysis:
    cpu = result.measurement("cpu")
    openmp = result.measurement("openmp")
    cuda = result.measurement("cuda")
    multi_gpu = result.measurement("cuda_multi_gpu")

    available = [item for item in result.measurements if item.available]
    # Multi-GPU is optional: a single-GPU machine can still produce a complete
    # CPU/OpenMP/CUDA comparison. Any wrong answer still blocks a recommendation.
    complete = all(result.measurement(name).available for name in CORE_IMPLEMENTATIONS)
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
    _add_speedup(speedups, "openmp_vs_cpu", cpu, openmp)
    _add_speedup(speedups, "cuda_vs_cpu", cpu, cuda)
    _add_speedup(speedups, "cuda_vs_openmp", openmp, cuda)
    _add_speedup(speedups, "cuda_multi_gpu_vs_cpu", cpu, multi_gpu)
    _add_speedup(speedups, "cuda_multi_gpu_vs_cuda", cuda, multi_gpu)

    scaling_efficiency: float | None = None
    if "cuda_multi_gpu_vs_cuda" in speedups:
        scaling_efficiency = round(
            cuda.latency_ms / multi_gpu.latency_ms / result.gpu_count,  # type: ignore[operator]
            3,
        )

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

    notes.append(f"latencies are medians of {result.repeats} trials")
    if result.transfer_included:
        notes.append("CUDA latency includes allocation and host/device transfers")
    else:
        notes.append("CUDA latency excludes host/device transfer overhead")
    if result.cuda_context_warmup_excluded:
        notes.append("one-time CUDA context initialization is excluded")
    if result.openmp_threads == 1:
        notes.append(
            "OpenMP ran with one thread, so openmp_vs_cpu does not measure "
            "parallel scaling"
        )
    if multi_gpu.available:
        notes.append(
            f"cuda_multi_gpu splits rows across {result.gpu_count} GPUs; each GPU "
            "receives the full right-hand matrix and GPUs do not communicate"
        )
    notes.extend(_telemetry_notes(result))

    return Analysis(
        fastest_implementation=fastest,  # type: ignore[arg-type]
        speedups=speedups,
        multi_gpu_scaling_efficiency=scaling_efficiency,
        correctness_passed=correctness_passed,
        complete_comparison=complete,
        multi_gpu_compared="cuda_multi_gpu_vs_cuda" in speedups,
        recommendation_allowed=complete and correctness_passed,
        notes=notes,
    )


def _telemetry_notes(result: BenchmarkResult) -> list[str]:
    telemetry = result.telemetry
    if telemetry is None:
        return ["GPU telemetry was not collected"]
    if not telemetry.available:
        return [f"GPU telemetry unavailable: {telemetry.error}"]

    notes = [
        "NVML utilization and power are averaged by the driver, so short phases "
        "can read lower than the true load"
    ]
    for phase in telemetry.phases:
        if phase.phase != "idle" and phase.sample_count < telemetry.min_phase_samples:
            notes.append(
                f"{phase.phase} telemetry on GPU {phase.gpu_index} has "
                f"{phase.sample_count} samples, too few to summarize"
            )
    return notes


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
