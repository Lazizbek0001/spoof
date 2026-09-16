from __future__ import annotations

import asyncio
import base64
import binascii
import secrets
import time
from concurrent.futures import Executor

import cv2
import numpy as np
from fastapi import WebSocket

# Import GPU configuration before DeepFace/TensorFlow.
from face_util import gpu_utils as _gpu_utils
from face_util.anti_spoof_infer import predict_antispoof_score

from deepface import DeepFace


def warmup_antispoof() -> None:
    try:
        dummy = np.zeros(
            (160, 160, 3),
            dtype=np.uint8,
        )

        predict_antispoof_score(dummy)

    except Exception as exc:
        print(
            f"[warmup] anti-spoof failed: {exc}"
        )


def warmup_deepface() -> None:
    try:
        DeepFace.represent(
            img_path=np.zeros(
                (160, 160, 3),
                dtype=np.uint8,
            ),
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False,
        )

    except Exception as exc:
        print(
            f"[warmup] DeepFace failed: {exc}"
        )


def decode_b64_image(
    data: str,
) -> bytes | None:
    if not data:
        return None

    data = data.strip()

    # Support:
    # data:image/jpeg;base64,...
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]

    try:
        return base64.b64decode(data)

    except (
        ValueError,
        binascii.Error,
    ):
        return None


def decode_bgr(
    image_bytes: bytes,
) -> np.ndarray | None:
    if not image_bytes:
        return None

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )


def run_liveness_on_frame(
    image_bytes: bytes,
) -> dict:
    image = decode_bgr(image_bytes)

    if image is None:
        raise ValueError(
            "cannot_decode_image"
        )

    started_at = time.perf_counter()

    is_real, score, reason = (
        predict_antispoof_score(image)
    )

    latency_ms = (
        time.perf_counter()
        - started_at
    ) * 1000

    # Infrastructure/model exceptions should not
    # be counted as fake biometric frames.
    if reason.startswith(
        "antispoof_exception:"
    ):
        raise RuntimeError(reason)

    return {
        "is_real": bool(is_real),
        "score": float(score),
        "reason": reason,
        "latency_ms": round(
            latency_ms,
            2,
        ),
    }


async def process_liveness_frame(
    *,
    image_bytes: bytes,
    executor: Executor,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> dict:
    loop = asyncio.get_running_loop()

    # Prevent an unlimited executor queue if many
    # clients connect simultaneously.
    async with semaphore:
        future = loop.run_in_executor(
            executor,
            run_liveness_on_frame,
            image_bytes,
        )

        return await asyncio.wait_for(
            future,
            timeout=timeout,
        )


async def check_api_key(
    websocket: WebSocket,
    expected_api_key: str,
) -> bool:
    client_api_key = (
        websocket.headers.get("x-api-key")
    )

    if not client_api_key:
        await websocket.close(
            code=1008,
            reason="API key missing",
        )
        return False

    if not secrets.compare_digest(
        client_api_key,
        expected_api_key,
    ):
        await websocket.close(
            code=1008,
            reason="Invalid API key",
        )
        return False

    return True


def get_remaining_time(
    loop: asyncio.AbstractEventLoop,
    deadline: float,
) -> float:
    return max(
        0.0,
        deadline - loop.time(),
    )


async def send_timeout(
    websocket: WebSocket,
    *,
    max_seconds: float,
) -> None:
    await websocket.send_json(
        {
            "type": "final",
            "ok": False,
            "reason": "connection_timeout",
            "message": (
                "Liveness session exceeded "
                f"{max_seconds:.0f} seconds."
            ),
        }
    )


async def send_ack(
    websocket: WebSocket,
    *,
    frame_index: int,
    result: dict,
    decision: dict,
    elapsed: float,
    remaining: float,
) -> None:
    await websocket.send_json(
        {
            "type": "ack",
            "frame_index": frame_index,
            "is_real": result["is_real"],
            "score": result["score"],
            "reason": result["reason"],
            "latency_ms": result[
                "latency_ms"
            ],
            "elapsed_seconds": round(
                elapsed,
                2,
            ),
            "remaining_seconds": round(
                remaining,
                2,
            ),
            "decision": decision,
        }
    )


async def send_final(
    websocket: WebSocket,
    *,
    ok: bool,
    reason: str,
    decision: dict,
    elapsed: float,
) -> None:
    await websocket.send_json(
        {
            "type": "final",
            "ok": ok,
            "reason": reason,
            "elapsed_seconds": round(
                elapsed,
                2,
            ),
            "decision": decision,
        }
    )