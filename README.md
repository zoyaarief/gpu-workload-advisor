# GPU Workload Advisor

A deliberately small AI agent that benchmarks square `float32` matrix
multiplication using sequential CPU, OpenMP, single-GPU CUDA, and multi-GPU CUDA
implementations. It repeats every measurement, samples GPU utilization, memory,
temperature, and power through NVML while the benchmark runs, validates the native
output, calculates speedups and multi-GPU scaling efficiency in Python, and asks an
LLM to explain only the measured evidence.

This MVP has no frontend, RAG system, vector database, arbitrary shell tool, or
multi-agent framework.

## What happens on a request

```text
Natural-language request
        ↓
LLM selects run_matrix_benchmark(matrix_size)
        ↓
Python validates the single integer argument
        ↓
NVML sampler starts ──────────────┐
        ↓                         │ samples every GPU
Fixed native executable runs CPU, │ in the background
OpenMP, 1-GPU, and multi-GPU      │
trials, each with a timing window │
        ↓                         │
NVML sampler stops ───────────────┘
        ↓
Python validates JSON, correctness, speedups, and
attributes telemetry samples to each GPU phase
        ↓
LLM explains those supplied facts
        ↓
API returns structured data and a Markdown report
```

The LLM never constructs a command. `app/benchmark.py` always executes one configured
binary with `shell=False`, a validated integer size, and the server-configured trial
count. The LLM cannot change the trial count.

## Project layout

```text
app/
  main.py       FastAPI routes and error mapping
  schemas.py    Request, benchmark, analysis, and response contracts
  benchmark.py  Safe native-process wrapper
  telemetry.py  Background NVML sampler and per-phase summaries
  analyzer.py   Deterministic speedup, scaling, and correctness analysis
  agent.py      One-tool Groq Responses API loop
  report.py     Reproducible Markdown report
native/
  matrix_benchmark.cu  CPU, OpenMP, 1-GPU and multi-GPU CUDA, timing, correctness
  CMakeLists.txt
tests/          GPU-independent unit and API tests
```

## Measurement policy

- CPU and OpenMP use the same row-oriented multiplication loop.
- CUDA uses a simple educational kernel rather than cuBLAS.
- CUDA latency includes allocation, input transfers, kernel execution, output
  transfer, and deallocation.
- One-time CUDA context initialization is warmed up and excluded on every GPU.
- Each implementation runs `BENCHMARK_REPEATS` trials (default 5). The reported
  latency is the median; the report also shows the min–max spread. Python checks that
  the reported median matches the trials.
- OpenMP and CUDA outputs are compared with the sequential CPU result using a
  floating-point tolerance on every trial.
- The benchmark reports the thread count of a real OpenMP parallel region. If the
  build silently drops OpenMP, it reports 1 and the report flags it.
- The first CUDA trial in a fresh process can include one-time costs, such as JIT
  compiling the kernel for a GPU the build did not target. Medians keep this out of
  the reported latency, and the min–max column still shows it.
- If CUDA is unavailable or any correctness check fails, the analyzer refuses to
  make a complete CPU/OpenMP/CUDA recommendation.

### Multi-GPU

- Output rows are split evenly across every visible GPU, with one host thread per
  GPU. Each GPU receives its slice of the left matrix and a full copy of the right
  matrix, runs the same kernel, and copies its rows back.
- GPUs never communicate with each other: no peer-to-peer, NVLink, or NCCL. Per-GPU
  allocation and transfers are timed, the same as the single-GPU run.
- Scaling efficiency = (single-GPU latency ÷ multi-GPU latency) ÷ GPU count. A value
  of 1.0 is perfect linear scaling. Each GPU still copies the full right matrix, so
  expect efficiency to rise with matrix size.
- With fewer than two GPUs, the multi-GPU phase is reported as unavailable, and the
  CPU/OpenMP/CUDA comparison still stands.

### GPU telemetry

- `app/telemetry.py` samples every GPU through NVML (`nvidia-ml-py`) every
  `TELEMETRY_INTERVAL_MS` (default 50 ms), plus a short idle baseline before the run.
- The native benchmark reports a wall-clock window for each implementation, and
  samples are attributed to the single-GPU and multi-GPU windows. Per-GPU usage
  during the multi-GPU window shows whether every GPU actually did work.
- The runner sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` so CUDA and NVML number GPUs the
  same way.
- NVML averages utilization and power over its own sampling period, so short phases
  can read low. Phases with fewer than two samples are not summarized.
- Telemetry is best effort. Without NVML (for example, on macOS), the benchmark still
  runs and the result says why telemetry is unavailable.

These choices make overhead visible and the report honest. They do not measure the
best possible GEMM performance; a cuBLAS comparison is outside this MVP.

## Measured Kaggle results

### 2 × Tesla T4 (2026-09-16)

At 2048 × 2048 with 5 trials per implementation (medians), all correctness checks
passed:

| Implementation | Median (ms) | Speedup vs CPU |
|---|---:|---:|
| CPU | 1661.241 | 1.0× |
| OpenMP (4 threads) | 815.333 | 2.0× |
| CUDA, 1 GPU | 68.244 | 24.3× |
| CUDA, 2 GPUs | 35.124 | 47.3× |

Two GPUs were 1.943× faster than one, a scaling efficiency of 0.971. Across the size
sweep, efficiency rose from 0.394 at 256 to 0.622 at 512, 0.940 at 1024, and 0.971 at
2048. NVML showed only GPU 0 working during the single-GPU run and both GPUs drawing
power during the two-GPU run. Per-trial values, telemetry, and caveats are in the
[T4 report](reports/kaggle_t4x2_results.md).

### Tesla P100 (2026-09-09)

A 1024 × 1024 run measured 225.588 ms for CPU and 11.944 ms for CUDA: 18.887× faster,
with allocation and host/device transfers included. See the
[P100 report](reports/kaggle_p100_results.md).

> **Corrections to the P100 run:**
> - **OpenMP was not parallel.** Its figure (224.435 ms, 1.005× CPU) was sequential:
>   `OpenMP::OpenMP_CXX` adds `-fopenmp` only to C++ sources, and
>   `matrix_benchmark.cu` is compiled as CUDA, so the OpenMP pragmas were silently
>   ignored. The build now forwards the flag, and the T4 run confirms 4 threads and a
>   2.0× speedup.
> - **The 256 × 256 CUDA time was probably mostly one-time cost.** It was a single
>   95.8 ms trial, which likely included first-launch cost such as JIT compilation,
>   not just allocation and transfers. With repeated trials on a T4, CUDA beat the CPU
>   even at 256 × 256.

## Prerequisites

- Python 3.10 or newer
- CMake 3.24 or newer
- A C++ compiler with OpenMP
- NVIDIA CUDA toolkit and a CUDA-capable GPU
- A free Groq API key for `/advise` only

The `/benchmark` endpoint does not call an LLM. The automated tests mock both the GPU
process and the LLM, so they run without CUDA hardware or an API key.

## No local NVIDIA GPU?

Use the ready-made
[Kaggle GPU notebook](notebooks/kaggle_gpu_demo.ipynb). Create a Kaggle notebook,
select **Settings → Accelerator → GPU T4 x2** (two GPUs, so the multi-GPU phase runs),
enable **Internet**, upload this notebook, and run the cells in order. Besides the
single measured run, it sweeps matrix sizes 256–2048 and saves a scaling table as
`scaling_sweep.md`. To test an unmerged branch, set `REPOSITORY_REF` in the clone
cell. It clones this repository and runs the complete project
on Kaggle's NVIDIA machine while your normal development and tests stay local.

Add `GROQ_API_KEY` through Kaggle's **Add-ons → Secrets** interface. Do not paste a
key into a code cell. Groq's free plan is sufficient for this demo. The notebook
saves the final report as `/kaggle/working/gpu_workload_report.md` so it can be
downloaded from the Output pane.

The final saved demonstration is available as
[Kaggle Version 2](https://www.kaggle.com/code/zoyaarief/gpu-workload-advisor-demo).

## 1. Build the native benchmark

```bash
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run it directly with a matrix size and an optional trial count (default 5, max 50):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/benchmark_runner 1024 5
```

It prints one JSON object. A CUDA measurement's `available` is false if the program
cannot access a CUDA-capable GPU, and `cuda_multi_gpu` is unavailable when fewer than
two GPUs are visible.

## 2. Install the API

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Export the values from `.env` in your shell. At minimum, set the key before using the
agent endpoint:

```bash
export GROQ_API_KEY="your-key"
export GROQ_MODEL="openai/gpt-oss-20b"
```

The agent uses Groq's OpenAI-compatible Responses API with one strict custom
function. The configured model selects the approved tool, while Python validates
and executes it locally. See the official
[Groq Responses API documentation](https://console.groq.com/docs/responses-api).

## 3. Start the service

```bash
python -m uvicorn app.main:app --reload --env-file .env
```

Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

Check the service:

```bash
curl http://127.0.0.1:8000/health
```

## 4. Run a structured benchmark

```bash
curl -X POST http://127.0.0.1:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"matrix_size": 1024}'
```

This returns the native measurements without calling the LLM.

## 5. Ask the advisor

```bash
curl -X POST http://127.0.0.1:8000/advise \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Benchmark 1024 x 1024 matrix multiplication and explain the result."}'
```

The response contains:

- `benchmark`: validated native measurements, per-trial latencies, and GPU telemetry
- `analysis`: deterministic speedups, multi-GPU scaling efficiency, and
  recommendation gate
- `explanation`: LLM explanation grounded in those fields
- `report`: a Markdown report with the command needed to reproduce the native run

## Run the tests

```bash
python -m pytest
```

The tests cover strict inputs, output validation (including median and trial-count
checks), safe process invocation, telemetry sampling and phase attribution with a
fake NVML, speedup and scaling math, correctness gating, tool-call restrictions, API
responses, and report wording.

## Run with Docker and NVIDIA Container Toolkit

```bash
docker build -t gpu-workload-advisor .
docker run --rm --gpus all \
  -p 8000:8000 \
  -e GROQ_API_KEY \
  -e GROQ_MODEL=openai/gpt-oss-20b \
  gpu-workload-advisor
```

Docker still requires a compatible host NVIDIA driver and NVIDIA Container Toolkit.
The toolkit also exposes NVML inside the container, so telemetry works there too.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Required for `/advise` |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Free-tier, tool-capable model |
| `BENCHMARK_EXECUTABLE` | `./build/benchmark_runner` | Fixed approved executable |
| `MAX_MATRIX_SIZE` | `2048` | Python-side safety limit |
| `BENCHMARK_TIMEOUT_SECONDS` | `120` | Native process timeout |
| `BENCHMARK_REPEATS` | `5` | Trials per implementation (1–50); latency is the median |
| `TELEMETRY_INTERVAL_MS` | `50` | NVML sampling period |

## How to read a speedup

`cuda_vs_cpu` is calculated as:

```text
CPU latency / CUDA latency
```

A value above `1.0` means CUDA was faster in that measured run. A value below `1.0`
means it was slower. The same rule applies to `openmp_vs_cpu`, `cuda_vs_openmp`,
`cuda_multi_gpu_vs_cpu`, and `cuda_multi_gpu_vs_cuda`.

`multi_gpu_scaling_efficiency` divides `cuda_multi_gpu_vs_cuda` by the GPU count, so
0.9 on two GPUs means a 1.8× speedup over one GPU.

## Expected errors

- `422`: invalid size, ambiguous/unsupported natural-language request, or unapproved
  tool arguments
- `503`: native executable not built or `GROQ_API_KEY` missing
- `502`: native benchmark failure, timeout, malformed output, or LLM service failure

Start with a small size such as `256`, then try `512` and `1024`. CPU multiplication
is cubic, so doubling the dimension increases the arithmetic work by roughly eight
times.
