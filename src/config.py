import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str

    max_connection_seconds: float = 30.0
    max_workers: int = int(os.getenv("max_workers", 6))

    window_size: int = 20
    min_real_frames: int = 16
    min_avg_score: float = 0.90
    required_real_ratio: float = 0.80


def load_settings() -> Settings:
    api_key = os.getenv("API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "API_KEY is not configured. "
            "Add API_KEY=... to your .env file."
        )

    return Settings(
        api_key=api_key,
    )


settings = load_settings()