import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = os.getenv("API_KEY", "").strip()
    reference_image: str = os.getenv("REFERENCE_IMAGE", "sample/5.jpg")

    max_connection_seconds: float = _float("MAX_CONNECTION_SECONDS", 60.0)
    max_sessions: int = _int("MAX_SESSIONS", 200)

    # Thread pools. CPU: decode, face bbox, crops, MediaPipe.
    # Recog: DeepFace / TensorFlow (more than 2 only adds contention).
    cpu_workers: int = _int("CPU_WORKERS", max(2, (os.cpu_count() or 4) - 1))
    recog_workers: int = _int("RECOG_WORKERS", 2)

    # Anti-spoof GPU micro-batching
    max_batch: int = _int("MAX_BATCH", 64)
    max_batch_wait_ms: float = _float("MAX_BATCH_WAIT_MS", 8.0)

    # Liveness window
    window_size: int = 20
    min_real_frames: int = 16
    min_avg_score: float = 0.90
    required_real_ratio: float = 0.80

    # Identity: consecutive matching frames needed, taken from the anti-spoof
    # frames. Recognition stops once reached.
    face_matches_required: int = 3
    # Challenge: consecutive frames in which the action is detected.
    action_matches_required: int = 2


settings = Settings()

if not settings.api_key:
    raise RuntimeError("API_KEY is not configured. Add API_KEY=... to your .env file.")