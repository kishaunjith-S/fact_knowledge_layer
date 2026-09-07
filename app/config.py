import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: str = field(default_factory=lambda: os.environ.get("FKL_DB_PATH", "data/facts.db"))
    cache_dir: str = field(default_factory=lambda: os.environ.get("FKL_CACHE_DIR", "data/cache"))
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY") or None)
    gemini_model: str = field(default_factory=lambda: os.environ.get("FKL_GEMINI_MODEL", "gemini-2.5-flash"))
    extraction_prompt_version: str = "v1"
    explanation_prompt_version: str = "v1"
    call_budget_per_document: int = field(default_factory=lambda: int(os.environ.get("FKL_CALL_BUDGET", "40")))
    explanation_budget_per_call: int = field(
        default_factory=lambda: int(os.environ.get("FKL_EXPLANATION_BUDGET", "20"))
    )
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.environ.get("FKL_RATE_LIMIT_RPM", "12")))
    magnitude_cap: float = 0.5
    vintage_gap_days: int = 90
    value_tolerance: float = 0.01
    value_tolerance_rounded: float = 0.05


def get_settings() -> Settings:
    return Settings()
