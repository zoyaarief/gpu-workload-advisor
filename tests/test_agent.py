from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent import (
    EXPLANATION_INSTRUCTIONS,
    GROQ_BASE_URL,
    AgentConfigurationError,
    GpuAdvisor,
    RequestUnderstandingError,
)
from app.config import Settings


class FakeRunner:
    def __init__(self, result) -> None:
        self.result = result
        self.received_size = None

    def run(self, matrix_size: int):
        self.received_size = matrix_size
        return self.result


class FakeResponses:
    def __init__(self, responses) -> None:
        self._responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = FakeResponses(responses)


def settings() -> Settings:
    return Settings(
        max_matrix_size=2048,
        benchmark_timeout_seconds=5,
        groq_model="test-model",
    )


def test_agent_executes_the_only_approved_tool_once(sample_result) -> None:
    function_call = SimpleNamespace(
        type="function_call",
        name="run_matrix_benchmark",
        arguments=json.dumps({"matrix_size": 1024}),
        call_id="call-1",
    )
    client = FakeClient(
        [
            SimpleNamespace(output=[function_call], output_text=""),
            SimpleNamespace(
                output=[],
                output_text=(
                    "CUDA was fastest in this measured run. All checks passed, "
                    "and the CUDA time includes transfers."
                ),
            ),
        ]
    )
    runner = FakeRunner(sample_result)

    response = GpuAdvisor(runner, settings(), client=client).advise(
        "Benchmark 1024 x 1024 matrix multiplication"
    )

    assert runner.received_size == 1024
    assert response.analysis.speedups["cuda_vs_cpu"] == 12.0
    assert "CUDA was fastest" in response.explanation
    assert "120.000" in response.report
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0]["parallel_tool_calls"] is False
    assert "store" not in client.responses.calls[0]
    assert "store" not in client.responses.calls[1]
    assert "tools" not in client.responses.calls[1]


def test_agent_configures_the_groq_responses_client(monkeypatch, sample_result) -> None:
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return FakeClient([])

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("app.agent.OpenAI", fake_openai)

    GpuAdvisor(FakeRunner(sample_result), settings())

    assert captured == {"api_key": "test-key", "base_url": GROQ_BASE_URL}


def test_agent_requires_a_groq_api_key(monkeypatch, sample_result) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(AgentConfigurationError, match="GROQ_API_KEY"):
        GpuAdvisor(FakeRunner(sample_result), settings())


def test_explanation_prompt_describes_the_educational_cuda_kernel() -> None:
    assert "simple educational kernel" in EXPLANATION_INSTRUCTIONS
    assert "not cuBLAS" in EXPLANATION_INSTRUCTIONS
    assert "not a tiled" in EXPLANATION_INSTRUCTIONS


def test_agent_does_not_guess_a_missing_size(sample_result) -> None:
    client = FakeClient(
        [
            SimpleNamespace(
                output=[],
                output_text="Please provide one square matrix size.",
            )
        ]
    )
    runner = FakeRunner(sample_result)

    with pytest.raises(RequestUnderstandingError, match="provide"):
        GpuAdvisor(runner, settings(), client=client).advise(
            "Benchmark matrix multiplication"
        )

    assert runner.received_size is None


def test_agent_rejects_extra_tool_arguments(sample_result) -> None:
    function_call = SimpleNamespace(
        type="function_call",
        name="run_matrix_benchmark",
        arguments=json.dumps({"matrix_size": 1024, "command": "anything"}),
        call_id="call-1",
    )
    client = FakeClient(
        [SimpleNamespace(output=[function_call], output_text="")]
    )

    with pytest.raises(RequestUnderstandingError, match="only matrix_size"):
        GpuAdvisor(FakeRunner(sample_result), settings(), client=client).advise(
            "Benchmark 1024"
        )


def test_agent_rejects_a_size_guessed_by_the_model(sample_result) -> None:
    function_call = SimpleNamespace(
        type="function_call",
        name="run_matrix_benchmark",
        arguments=json.dumps({"matrix_size": 1024}),
        call_id="call-1",
    )
    client = FakeClient(
        [SimpleNamespace(output=[function_call], output_text="")]
    )
    runner = FakeRunner(sample_result)

    with pytest.raises(RequestUnderstandingError, match="must appear"):
        GpuAdvisor(runner, settings(), client=client).advise(
            "Benchmark matrix multiplication"
        )

    assert runner.received_size is None


def test_agent_rejects_non_square_dimensions(sample_result) -> None:
    function_call = SimpleNamespace(
        type="function_call",
        name="run_matrix_benchmark",
        arguments=json.dumps({"matrix_size": 1024}),
        call_id="call-1",
    )
    client = FakeClient(
        [SimpleNamespace(output=[function_call], output_text="")]
    )

    with pytest.raises(RequestUnderstandingError, match="square"):
        GpuAdvisor(FakeRunner(sample_result), settings(), client=client).advise(
            "Benchmark 1024 x 2048 matrix multiplication"
        )
