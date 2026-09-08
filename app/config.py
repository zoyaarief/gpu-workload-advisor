from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    benchmark_executable: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "BENCHMARK_EXECUTABLE",
                PROJECT_ROOT / "build" / "benchmark_runner",
            )
        ).resolve()
    )
    max_matrix_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_MATRIX_SIZE", "2048"))
    )
    benchmark_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("BENCHMARK_TIMEOUT_SECONDS", "120"))
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    )
