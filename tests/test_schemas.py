from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BenchmarkRequest, BenchmarkResult


@pytest.mark.parametrize("value", [0, -1, 2.5, "1024", True])
def test_matrix_size_is_a_strict_positive_integer(value) -> None:
    with pytest.raises(ValidationError):
        BenchmarkRequest(matrix_size=value)


def test_result_requires_each_implementation_exactly_once(sample_result) -> None:
    data = sample_result.model_dump()
    data["measurements"][2]["implementation"] = "cpu"

    with pytest.raises(ValidationError, match="exactly once"):
        BenchmarkResult.model_validate(data)
