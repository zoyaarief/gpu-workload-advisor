FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY native/ native/
RUN cmake -S native -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel

COPY pyproject.toml README.md ./
COPY app/ app/
RUN python3 -m pip install --no-cache-dir .

ENV BENCHMARK_EXECUTABLE=/app/build/benchmark_runner

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
