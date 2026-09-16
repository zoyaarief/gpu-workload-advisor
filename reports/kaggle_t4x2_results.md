# Kaggle 2× Tesla T4 Benchmark Results

Run date: 2026-09-16

Environment:

- Kaggle notebook, accelerator **GPU T4 x2**
- GPUs: 2 × Tesla T4, 15360 MiB each (NVML)
- CUDA compiler: NVIDIA 12.8.93, host compiler GNU 11.4.0, Python 3.12
- Commit: `3899165` on `multi-gpu-telemetry`
- OpenMP threads used (real parallel region): 4. The CPU's physical core count was
  not measured.
- Data type: float32
- CUDA allocation and host/device transfers: included
- One-time CUDA context initialization: excluded on every GPU
- Correctness reference: sequential CPU result, float tolerance, checked on every trial
- The notebook's 56 GPU-independent tests passed on Kaggle before the run.

## 2048 × 2048, 5 trials per implementation

| Implementation | Median (ms) | Min–max (ms) | Correct |
|---|---:|---:|---|
| CPU | 1661.241 | 1634.273–1741.512 | yes |
| OpenMP | 815.333 | 807.186–911.873 | yes |
| CUDA, 1 GPU | 68.244 | 55.582–188.435 | yes |
| CUDA, 2 GPUs | 35.124 | 27.013–35.632 | yes |

Trial latencies in run order (ms):

| Implementation | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 |
|---|---:|---:|---:|---:|---:|
| CPU | 1661.241 | 1634.273 | 1655.313 | 1695.098 | 1741.512 |
| OpenMP | 911.873 | 807.186 | 818.813 | 815.333 | 813.554 |
| CUDA, 1 GPU | 188.435 | 68.244 | 67.822 | 76.869 | 55.582 |
| CUDA, 2 GPUs | 35.630 | 35.632 | 35.124 | 35.003 | 27.013 |

### Calculated speedups

- OpenMP versus CPU: 2.038×
- CUDA (1 GPU) versus CPU: 24.343×
- CUDA (1 GPU) versus OpenMP: 11.947×
- CUDA (2 GPUs) versus CPU: 47.296×
- CUDA (2 GPUs) versus CUDA (1 GPU): 1.943×
- Multi-GPU scaling efficiency: 0.971 (1.943× ÷ 2 GPUs)

### GPU telemetry (NVML, sampled every 50 ms)

| Phase | GPU | Samples | Mean util (%) | Max util (%) | Peak memory (MiB) | Max temp (°C) | Mean power (W) | Max power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Idle baseline | 0 | 3 | 0 | 0 | 448.3 | 48 | 10.8 | 11.8 |
| Idle baseline | 1 | 3 | 0 | 0 | 448.3 | 47 | 10.2 | 10.5 |
| CUDA, 1 GPU | 0 | 8 | 33.5 | 86 | 601.2 | 51 | 40.2 | 58.8 |
| CUDA, 1 GPU | 1 | 8 | 0 | 0 | 451.2 | 46 | 10.4 | 11.6 |
| CUDA, 2 GPUs | 0 | 3 | 23.3 | 28 | 585.2 | 52 | 54.2 | 84.1 |
| CUDA, 2 GPUs | 1 | 3 | 3 | 3 | 585.2 | 49 | 33.7 | 45.5 |

What the telemetry shows:

- **Single-GPU run:** only GPU 0 worked. It reached 86% utilization and 58.8 W,
  while GPU 1 stayed at idle power.
- **Two-GPU run:** both GPUs worked. GPU 1 drew up to 45.5 W against a 10–12 W idle
  baseline, and both GPUs allocated the same memory. GPU 1's 3% utilization
  under-reads: the whole phase lasted about 194 ms, and NVML averages utilization
  over a longer period.
- **Memory matches the row split.** A 2048 × 2048 float32 matrix is 16 MiB. The
  single-GPU run holds three full matrices (48 MiB). Each GPU in the two-GPU run
  holds half of the left matrix, the full right matrix, and half of the output
  (32 MiB). The measured difference on GPU 0, 601.2 − 585.2 = 16 MiB, matches.

## Multi-GPU scaling sweep (median of 5 trials)

| Size | CPU (ms) | OpenMP (ms) | CUDA 1 GPU (ms) | CUDA 2 GPUs (ms) | 2 GPUs vs 1 | Scaling efficiency | All correct |
|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | 3.143 | 1.493 | 0.655 | 0.831 | 0.788× | 0.394 | yes |
| 512 | 26.155 | 12.662 | 1.984 | 1.595 | 1.244× | 0.622 | yes |
| 1024 | 216.801 | 104.865 | 13.319 | 7.088 | 1.879× | 0.940 | yes |
| 2048 | 1661.241 | 815.333 | 68.244 | 35.124 | 1.943× | 0.971 | yes |

Two GPUs were slower than one at 256 × 256 and close to linear at 2048 × 2048. Some
per-GPU costs don't shrink when work is split: a host thread, context and allocation
setup, and a full copy of the right matrix (which grows only with the square of the
size). The arithmetic that is split grows with the cube of the size, so it dominates
at larger sizes. That explanation fits the trend but was not measured separately.

## First-trial cost

The first single-GPU trial at 2048 took 188 ms; the other four took 56–77 ms. This
run was the notebook's first CUDA process. nvcc also warned that the build targets
compute capabilities older than the T4's 7.5, so the kernel was probably
JIT-compiled for the T4 on its first launch.

This likely explains the older P100 result, where CUDA took 95.8 ms at 256 × 256
from a single trial in the first CUDA process. Here, with warm repeated trials, one
T4 beat the CPU at 256 × 256 (0.655 ms vs 3.143 ms). Medians keep this one-time
cost out of the reported latency, and the min–max column still shows it.

## What this does not prove

- Five trials per size on one Kaggle machine; other GPUs, CPUs, interconnects, and
  sizes can behave differently.
- The CUDA kernel is an educational per-element kernel, not cuBLAS.
- The multi-GPU run uses no peer-to-peer, NVLink, or NCCL communication. It shows
  data-parallel row splitting, not collective scaling.
- The JIT explanation for the first-trial cost is an inference from the compiler
  warning and the timing pattern, not a separate measurement.
- This run had no LLM explanation because no Groq secret was attached. The
  deterministic report was used.

## Reproduce

Open [the notebook](../notebooks/kaggle_gpu_demo.ipynb) on Kaggle with **GPU T4 x2**
and run all cells, or run directly on a two-GPU machine:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/benchmark_runner 2048 5
```
