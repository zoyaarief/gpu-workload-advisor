from __future__ import annotations

from app.schemas import Analysis, BenchmarkResult


def generate_report(
    benchmark: BenchmarkResult,
    analysis: Analysis,
    explanation: str,
) -> str:
    measurement_rows = []
    for item in benchmark.measurements:
        latency = f"{item.latency_ms:.3f}" if item.latency_ms is not None else "N/A"
        correctness = (
            "pass" if item.correct is True else "fail" if item.correct is False else "N/A"
        )
        status = "available" if item.available else f"unavailable: {item.error}"
        measurement_rows.append(
            f"| {item.implementation.upper()} | {latency} | {correctness} | {status} |"
        )

    if analysis.speedups:
        speedup_lines = "\n".join(
            f"- `{name}`: {value:.3f}×" for name, value in analysis.speedups.items()
        )
    else:
        speedup_lines = "- No valid speedups could be calculated."

    if analysis.recommendation_allowed and analysis.fastest_implementation:
        verdict = (
            f"The fastest correct implementation in this run was "
            f"**{analysis.fastest_implementation.upper()}**."
        )
    else:
        verdict = (
            "No CPU/OpenMP/CUDA recommendation is valid because the comparison "
            "was incomplete or a correctness check failed."
        )

    transfer_policy = "included" if benchmark.transfer_included else "excluded"
    warmup_policy = (
        "excluded" if benchmark.cuda_context_warmup_excluded else "included"
    )

    return f"""# GPU Workload Advisor Report

## Workload

- Operation: square float32 matrix multiplication
- Matrix size: {benchmark.matrix_size} × {benchmark.matrix_size}
- GPU: {benchmark.gpu_name}
- OpenMP maximum threads: {benchmark.openmp_threads}

## Measured results

| Implementation | Latency (ms) | Correctness | Status |
|---|---:|---|---|
{chr(10).join(measurement_rows)}

## Calculated speedups

{speedup_lines}

## Recommendation

{verdict}

{explanation.strip()}

## Measurement policy and limitations

- CUDA allocation and host/device transfers: {transfer_policy}.
- One-time CUDA context initialization: {warmup_policy}.
- Correctness is checked against the sequential CPU result with a float tolerance.
- These results describe this matrix size, implementation, build, and machine only.
- A single run does not establish performance for other hardware or workloads.

## Reproduce

```bash
./build/benchmark_runner {benchmark.matrix_size}
```
"""
