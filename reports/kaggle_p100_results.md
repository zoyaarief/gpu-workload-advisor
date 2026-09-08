# Kaggle Tesla P100 Benchmark Results

Run date: 2026-09-08

Environment reported by the benchmark:

- GPU: Tesla P100-PCIE-16GB
- OpenMP maximum threads: 4
- Data type: float32
- CUDA allocation and host/device transfers: included
- One-time CUDA context initialization: excluded
- Correctness reference: sequential CPU result with floating-point tolerance

## Measurements

| Matrix size | CPU (ms) | OpenMP (ms) | CUDA (ms) | Fastest | All correct |
|---:|---:|---:|---:|---|---|
| 256 × 256 | 2.884 | 3.984 | 95.786 | CPU | Yes |
| 1024 × 1024 | 204.726 | 200.821 | 11.548 | CUDA | Yes |

## Calculated speedups

For 256 × 256:

- OpenMP versus CPU: 0.724×
- CUDA versus CPU: 0.030×
- CUDA versus OpenMP: 0.042×

For 1024 × 1024:

- OpenMP versus CPU: 1.019×
- CUDA versus CPU: 17.729×
- CUDA versus OpenMP: 17.391×

## Evidence-backed recommendation

For this implementation and Kaggle machine, CPU was the best choice at 256 × 256,
where CUDA setup, allocation, and transfer overhead dominated the small workload. At
1024 × 1024, CUDA was the best measured choice and completed the operation 17.729×
faster than the sequential CPU result even with allocation and transfers included.

This is consistent with a larger workload providing enough parallel arithmetic to
amortize GPU overhead. That explanation is an interpretation of the measured pattern,
not a separately measured cause.

## What this does not prove

- Each size was measured once; the MVP does not report statistical variation.
- The CUDA implementation is an educational kernel, not cuBLAS.
- Results do not generalize to other GPUs, CPUs, matrix sizes, or applications.
- The benchmark does not identify an exact CPU model.
- The one-time CUDA context initialization is not included.

## Reproduce

```bash
./build/benchmark_runner 256
./build/benchmark_runner 1024
```
