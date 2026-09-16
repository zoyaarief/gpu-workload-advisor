from __future__ import annotations

from app.schemas import Analysis, BenchmarkResult, GpuTelemetry


IMPLEMENTATION_LABELS = {
    "cpu": "CPU",
    "openmp": "OpenMP",
    "cuda": "CUDA (1 GPU)",
    "cuda_multi_gpu": "CUDA (multi-GPU)",
    "idle": "Idle baseline",
}


def generate_report(
    benchmark: BenchmarkResult,
    analysis: Analysis,
    explanation: str,
) -> str:
    measurement_rows = []
    for item in benchmark.measurements:
        latency = f"{item.latency_ms:.3f}" if item.latency_ms is not None else "N/A"
        spread = (
            f"{min(item.trial_latencies_ms):.3f}–{max(item.trial_latencies_ms):.3f}"
            if item.trial_latencies_ms
            else "N/A"
        )
        correctness = (
            "pass" if item.correct is True else "fail" if item.correct is False else "N/A"
        )
        status = "available" if item.available else f"unavailable: {item.error}"
        measurement_rows.append(
            f"| {IMPLEMENTATION_LABELS[item.implementation]} | {latency} | {spread} "
            f"| {correctness} | {status} |"
        )

    if analysis.speedups:
        speedup_lines = "\n".join(
            f"- `{name}`: {value:.3f}×" for name, value in analysis.speedups.items()
        )
    else:
        speedup_lines = "- No valid speedups could be calculated."
    if analysis.multi_gpu_scaling_efficiency is not None:
        speedup_lines += (
            f"\n- Multi-GPU scaling efficiency: "
            f"{analysis.multi_gpu_scaling_efficiency:.3f} "
            f"(multi-GPU speedup over one GPU ÷ {benchmark.gpu_count} GPUs; "
            "1.000 is perfect linear scaling)"
        )

    if analysis.recommendation_allowed and analysis.fastest_implementation:
        verdict = (
            f"The fastest correct implementation in this run was "
            f"**{IMPLEMENTATION_LABELS[analysis.fastest_implementation]}**."
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
    openmp_warning = (
        " (OpenMP did not parallelize; check that the build passes the OpenMP flag)"
        if benchmark.openmp_threads == 1
        else ""
    )

    return f"""# GPU Workload Advisor Report

## Workload

- Operation: square float32 matrix multiplication
- Matrix size: {benchmark.matrix_size} × {benchmark.matrix_size}
- GPU: {benchmark.gpu_name} (CUDA devices visible: {benchmark.gpu_count})
- OpenMP threads used: {benchmark.openmp_threads}{openmp_warning}
- Trials per implementation: {benchmark.repeats}

## Measured results

| Implementation | Median (ms) | Min–max (ms) | Correctness | Status |
|---|---:|---:|---|---|
{chr(10).join(measurement_rows)}

## Calculated speedups

{speedup_lines}

## GPU telemetry

{_telemetry_section(benchmark.telemetry)}

## Recommendation

{verdict}

{explanation.strip()}

## Measurement policy and limitations

- CUDA allocation and host/device transfers: {transfer_policy}.
- One-time CUDA context initialization: {warmup_policy}.
- Each latency is the median of {benchmark.repeats} trials.
- The multi-GPU run splits rows evenly across GPUs, copies the full right-hand
  matrix to every GPU, and uses no peer-to-peer or collective communication.
- Correctness is checked against the sequential CPU result with a float tolerance
  on every trial.
- These results describe this matrix size, implementation, build, and machine only.
- {benchmark.repeats} trials on one machine do not establish performance for other
  hardware or workloads.

## Reproduce

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/benchmark_runner {benchmark.matrix_size} {benchmark.repeats}
```
"""


def _telemetry_section(telemetry: GpuTelemetry | None) -> str:
    if telemetry is None:
        return "GPU telemetry was not collected."
    if not telemetry.available:
        return f"GPU telemetry was unavailable: {telemetry.error}"

    devices = "\n".join(
        f"- GPU {device.index}: {device.name}, {device.memory_total_mib:.0f} MiB"
        for device in telemetry.devices
    )
    rows = []
    for phase in telemetry.phases:
        values = [
            phase.utilization_mean_percent,
            phase.utilization_max_percent,
            phase.memory_used_max_mib,
            phase.temperature_max_c,
            phase.power_mean_w,
            phase.power_max_w,
        ]
        cells = " | ".join("N/A" if value is None else f"{value:g}" for value in values)
        rows.append(
            f"| {IMPLEMENTATION_LABELS[phase.phase]} | {phase.gpu_index} "
            f"| {phase.sample_count} | {cells} |"
        )

    return f"""Sampled with NVML every {telemetry.sample_interval_ms:g} ms. GPU indices use
PCI bus order for both NVML and CUDA. Phases with fewer than
{telemetry.min_phase_samples} samples are not summarized. NVML averages utilization
and power over its own sampling period, so short phases can read low.

{devices}

| Phase | GPU | Samples | Mean util (%) | Max util (%) | Peak memory (MiB) | Max temp (°C) | Mean power (W) | Max power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}"""
