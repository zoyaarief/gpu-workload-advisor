# GPU Workload Advisor

A deliberately small AI agent that benchmarks square `float32` matrix
multiplication using sequential CPU, OpenMP, and CUDA implementations. It validates
the native output, calculates speedups in Python, and asks an LLM to explain only the
measured evidence.

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
Fixed native executable runs CPU, OpenMP, and CUDA
        ↓
Python validates JSON, correctness, and speedups
        ↓
LLM explains those supplied facts
        ↓
API returns structured data and a Markdown report
```

The LLM never constructs a command. `app/benchmark.py` always executes one configured
binary with `shell=False` and a validated integer argument.

## Project layout

```text
app/
  main.py       FastAPI routes and error mapping
  schemas.py    Request, benchmark, analysis, and response contracts
  benchmark.py  Safe native-process wrapper
  analyzer.py   Deterministic speedup and correctness analysis
  agent.py      One-tool Groq Responses API loop
  report.py     Reproducible Markdown report
native/
  matrix_benchmark.cu  CPU, OpenMP, CUDA, timing, and correctness
  CMakeLists.txt
tests/          GPU-independent unit and API tests
```

## Measurement policy

- CPU and OpenMP use the same row-oriented multiplication loop.
- CUDA uses a simple educational kernel rather than cuBLAS.
- CUDA latency includes allocation, input transfers, kernel execution, output
  transfer, and deallocation.
- One-time CUDA context initialization is warmed up and excluded.
- OpenMP and CUDA outputs are compared with the sequential CPU result using a
  floating-point tolerance.
- Each implementation is timed once in this MVP. Results apply only to that run,
  matrix size, build, and machine.
- If CUDA is unavailable or any correctness check fails, the analyzer refuses to
  make a complete CPU/OpenMP/CUDA recommendation.

These choices make overhead visible and the report honest. They do not measure the
best possible GEMM performance; a cuBLAS comparison and repeated trials are outside
this MVP.

## Measured Kaggle result

On a Kaggle Tesla P100, the validated 1024 × 1024 run measured 202.259 ms for CPU,
202.252 ms for OpenMP, and 11.209 ms for CUDA. All correctness checks passed, and
CUDA was 18.044× faster than CPU with allocation and host/device transfers included.
At 256 × 256, CPU was faster because the included CUDA overhead dominated the small
workload. See the full [measured report](reports/kaggle_p100_results.md) for the raw
values, calculation policy, and limitations.

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
select **Settings → Accelerator → GPU**, enable **Internet**, upload this notebook,
and run the cells in order. It clones this repository and runs the complete project
on Kaggle's NVIDIA machine while your normal development and tests stay local.

Add `GROQ_API_KEY` through Kaggle's **Add-ons → Secrets** interface. Do not paste a
key into a code cell. Groq's free plan is sufficient for this demo. The notebook
saves the final report as `/kaggle/working/gpu_workload_report.md` so it can be
downloaded from the Output pane.

## 1. Build the native benchmark

```bash
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run it directly:

```bash
./build/benchmark_runner 1024
```

It prints one JSON object. `cuda.available` will be false if the compiled program
cannot access a CUDA-capable GPU.

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

- `benchmark`: validated native measurements
- `analysis`: deterministic speedups and recommendation gate
- `explanation`: LLM explanation grounded in those fields
- `report`: a Markdown report with the command needed to reproduce the native run

## Run the tests

```bash
python -m pytest
```

The tests cover strict inputs, output validation, safe process invocation, speedup
math, correctness gating, tool-call restrictions, API responses, and report wording.

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

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Required for `/advise` |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Free-tier, tool-capable model |
| `BENCHMARK_EXECUTABLE` | `./build/benchmark_runner` | Fixed approved executable |
| `MAX_MATRIX_SIZE` | `2048` | Python-side safety limit |
| `BENCHMARK_TIMEOUT_SECONDS` | `120` | Native process timeout |

## How to read a speedup

`cuda_vs_cpu` is calculated as:

```text
CPU latency / CUDA latency
```

A value above `1.0` means CUDA was faster in that measured run. A value below `1.0`
means it was slower. The same rule applies to `openmp_vs_cpu` and
`cuda_vs_openmp`.

## Expected errors

- `422`: invalid size, ambiguous/unsupported natural-language request, or unapproved
  tool arguments
- `503`: native executable not built or `GROQ_API_KEY` missing
- `502`: native benchmark failure, timeout, malformed output, or LLM service failure

Start with a small size such as `256`, then try `512` and `1024`. CPU multiplication
is cubic, so doubling the dimension increases the arithmetic work by roughly eight
times.
