from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from app.analyzer import analyze
from app.benchmark import BenchmarkRunner
from app.config import Settings
from app.report import generate_report
from app.schemas import AdviceResponse


TOOL_NAME = "run_matrix_benchmark"

AGENT_INSTRUCTIONS = """You are a narrow GPU matrix-multiplication advisor.
You have exactly one approved tool: run_matrix_benchmark.
Use it only for a request to benchmark square matrix multiplication when the user
provides one unambiguous positive integer matrix size. Never guess a size and never
invent a tool or command. If the request is unsupported or lacks a size, do not call
the tool; briefly say what is missing. Call the tool at most once.
"""

EXPLANATION_INSTRUCTIONS = """Explain the benchmark evidence supplied by the tool.
Use only the provided measurements and calculated speedups. State which measured
implementation was fastest, whether all correctness checks passed, whether CUDA
transfer/allocation time was included, and why the result may have occurred. Treat
performance reasons as cautious interpretation rather than measured fact. If the
comparison is incomplete or correctness failed, do not recommend an implementation.
State that one run does not prove performance for other sizes or hardware. Be concise.
"""


class AgentError(Exception):
    """Base class for agent failures safe to map to an API response."""


class AgentConfigurationError(AgentError):
    pass


class RequestUnderstandingError(AgentError):
    pass


class LLMServiceError(AgentError):
    pass


class GpuAdvisor:
    def __init__(
        self,
        runner: BenchmarkRunner,
        settings: Settings,
        client: Any | None = None,
    ) -> None:
        self._runner = runner
        self._settings = settings
        if client is not None:
            self._client = client
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise AgentConfigurationError(
                    "OPENAI_API_KEY is required for the /advise endpoint"
                )
            self._client = OpenAI()

    def advise(self, prompt: str) -> AdviceResponse:
        initial_input: list[Any] = [{"role": "user", "content": prompt}]
        try:
            first_response = self._client.responses.create(
                model=self._settings.openai_model,
                instructions=AGENT_INSTRUCTIONS,
                input=initial_input,
                tools=[self._tool_definition()],
                tool_choice="auto",
                parallel_tool_calls=False,
                store=False,
            )
        except Exception as error:
            raise LLMServiceError("the LLM request failed") from error

        function_calls = [
            item
            for item in first_response.output
            if getattr(item, "type", None) == "function_call"
        ]
        if len(function_calls) != 1:
            model_message = getattr(first_response, "output_text", "").strip()
            raise RequestUnderstandingError(
                model_message
                or "include one matrix size, for example: benchmark 1024 × 1024"
            )

        function_call = function_calls[0]
        if getattr(function_call, "name", None) != TOOL_NAME:
            raise RequestUnderstandingError("the model requested an unapproved tool")

        matrix_size = self._parse_tool_arguments(function_call.arguments)
        self._validate_size_is_in_prompt(prompt, matrix_size)
        benchmark = self._runner.run(matrix_size)
        analysis = analyze(benchmark)
        tool_result = json.dumps(
            {
                "benchmark": benchmark.model_dump(mode="json"),
                "analysis": analysis.model_dump(mode="json"),
            }
        )

        follow_up_input = [
            *initial_input,
            *list(first_response.output),
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": tool_result,
            },
        ]
        try:
            final_response = self._client.responses.create(
                model=self._settings.openai_model,
                instructions=EXPLANATION_INSTRUCTIONS,
                input=follow_up_input,
                store=False,
            )
        except Exception as error:
            raise LLMServiceError("the LLM explanation request failed") from error

        explanation = final_response.output_text.strip()
        if not explanation:
            raise LLMServiceError("the LLM returned an empty explanation")

        return AdviceResponse(
            benchmark=benchmark,
            analysis=analysis,
            explanation=explanation,
            report=generate_report(benchmark, analysis, explanation),
        )

    def _tool_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": TOOL_NAME,
            "description": (
                "Run the approved sequential CPU, OpenMP, and CUDA square "
                "matrix-multiplication benchmark."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "matrix_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._settings.max_matrix_size,
                        "description": "The N in an N by N square matrix.",
                    }
                },
                "required": ["matrix_size"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _parse_tool_arguments(raw_arguments: str) -> int:
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError) as error:
            raise RequestUnderstandingError(
                "the model returned invalid tool arguments"
            ) from error

        if set(arguments) != {"matrix_size"}:
            raise RequestUnderstandingError("the tool accepts only matrix_size")
        matrix_size = arguments["matrix_size"]
        if isinstance(matrix_size, bool) or not isinstance(matrix_size, int):
            raise RequestUnderstandingError("matrix_size must be an integer")
        return matrix_size

    @staticmethod
    def _validate_size_is_in_prompt(prompt: str, matrix_size: int) -> None:
        dimension_pairs = re.findall(r"(?<!\w)(\d+)\s*[x×]\s*(\d+)(?!\w)", prompt)
        if dimension_pairs:
            parsed_pairs = {(int(left), int(right)) for left, right in dimension_pairs}
            if parsed_pairs != {(matrix_size, matrix_size)}:
                raise RequestUnderstandingError(
                    "provide one unambiguous square matrix size"
                )
            return

        numeric_values = {int(value) for value in re.findall(r"(?<!\w)\d+(?!\w)", prompt)}
        if numeric_values != {matrix_size}:
            raise RequestUnderstandingError(
                "the tool's matrix size must appear unambiguously in the request"
            )
