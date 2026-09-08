from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app, get_advisor, get_runner
from app.schemas import AdviceResponse


class FakeRunner:
    def __init__(self, result) -> None:
        self.result = result

    def run(self, matrix_size: int):
        assert matrix_size == 1024
        return self.result


class FakeAdvisor:
    def __init__(self, result) -> None:
        self.result = result

    def advise(self, prompt: str) -> AdviceResponse:
        assert "1024" in prompt
        return self.result


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_benchmark_endpoint_returns_native_measurements(sample_result) -> None:
    app.dependency_overrides[get_runner] = lambda: FakeRunner(sample_result)
    try:
        response = TestClient(app).post(
            "/benchmark", json={"matrix_size": 1024}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["measurements"][2]["implementation"] == "cuda"


def test_benchmark_endpoint_rejects_extra_input(sample_result) -> None:
    app.dependency_overrides[get_runner] = lambda: FakeRunner(sample_result)
    try:
        response = TestClient(app).post(
            "/benchmark", json={"matrix_size": 1024, "command": "whoami"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_advice_endpoint_returns_report(sample_result) -> None:
    from app.analyzer import analyze
    from app.report import generate_report

    explanation = "CUDA was fastest for this measured workload."
    analysis = analyze(sample_result)
    advice = AdviceResponse(
        benchmark=sample_result,
        analysis=analysis,
        explanation=explanation,
        report=generate_report(sample_result, analysis, explanation),
    )
    app.dependency_overrides[get_advisor] = lambda: FakeAdvisor(advice)
    try:
        response = TestClient(app).post(
            "/advise",
            json={"prompt": "Benchmark 1024 x 1024 matrix multiplication"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["analysis"]["fastest_implementation"] == "cuda"
    assert response.json()["report"].startswith("# GPU Workload Advisor Report")
