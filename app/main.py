from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agent import (
    AgentConfigurationError,
    GpuAdvisor,
    LLMServiceError,
    RequestUnderstandingError,
)
from app.benchmark import (
    BenchmarkExecutionError,
    BenchmarkInputError,
    BenchmarkRunner,
    BenchmarkUnavailableError,
)
from app.config import Settings
from app.schemas import AdviceRequest, AdviceResponse, BenchmarkRequest, BenchmarkResult


app = FastAPI(
    title="GPU Workload Advisor",
    version="0.1.0",
    description="Evidence-backed CPU, OpenMP, and CUDA matrix benchmark advice.",
)

settings = Settings()
benchmark_runner = BenchmarkRunner(settings)


@app.exception_handler(AgentConfigurationError)
async def agent_configuration_error(
    _request: Request, error: AgentConfigurationError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


def get_runner() -> BenchmarkRunner:
    return benchmark_runner


def get_advisor(
    runner: Annotated[BenchmarkRunner, Depends(get_runner)],
) -> GpuAdvisor:
    return GpuAdvisor(runner, settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/benchmark", response_model=BenchmarkResult)
def run_benchmark(
    request: BenchmarkRequest,
    runner: Annotated[BenchmarkRunner, Depends(get_runner)],
) -> BenchmarkResult:
    try:
        return runner.run(request.matrix_size)
    except BenchmarkInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except BenchmarkUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except BenchmarkExecutionError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/advise", response_model=AdviceResponse)
def advise(
    request: AdviceRequest,
    advisor: Annotated[GpuAdvisor, Depends(get_advisor)],
) -> AdviceResponse:
    try:
        return advisor.advise(request.prompt)
    except (BenchmarkInputError, RequestUnderstandingError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (BenchmarkUnavailableError, AgentConfigurationError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (BenchmarkExecutionError, LLMServiceError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
