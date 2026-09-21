import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str

    max_connection_seconds: float = _env_float("MAX_CONNECTION_SECONDS", 60.0)

    # ---- Concurrency ---------------------------------------------------
    # CPU pool: JPEG decode, face bbox, screen heuristic, crops, MediaPipe.
    # Old env name `max_workers` is still honoured.
    cpu_workers: int = _env_int(
        "CPU_WORKERS",
        _env_int("max_workers", max(2, (os.cpu_count() or 4) - 1)),
    )
    # DeepFace / TensorFlow threads. TF serialises GPU work internally,
    # more than 2 only adds contention.
    recog_workers: int = _env_int("RECOG_WORKERS", 2)
    # Hard cap on simultaneous websocket sessions (extra ones are refused
    # instead of degrading everyone).
    max_sessions: int = _env_int("MAX_SESSIONS", 200)

    # ---- Anti-spoof GPU micro-batching -----------------------------------
    max_batch: int = _env_int("MAX_BATCH", 64)
    max_batch_wait_ms: float = _env_float("MAX_BATCH_WAIT_MS", 8.0)

    # ---- Liveness window -------------------------------------------------
    window_size: int = 20
    min_real_frames: int = 16
    min_avg_score: float = 0.90
    required_real_ratio: float = 0.80

    # ---- Face recognition ------------------------------------------------
    reference_image: str = os.getenv("REFERENCE_IMAGE", "sample/5.jpg")
    face_matches_required: int = 3
    action_matches_required: int = 2

    # ---- Cross-stage checks ----------------------------------------------
    # Anti-spoof also runs on the frames used for recognition, so a photo of
    # the victim cannot be shown after a live person passed stage 1.
    recheck_liveness_on_recognition: bool = _env_bool(
        "RECHECK_LIVENESS_ON_RECOGNITION", True
    )
    # The frame that completes the action must still match the reference,
    # so the person doing the action is the person that was recognised.
    verify_identity_on_action: bool = _env_bool(
        "VERIFY_IDENTITY_ON_ACTION", True
    )


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