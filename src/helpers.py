from __future__ import annotations

import asyncio
import base64
import binascii
import json
import secrets
import time
from concurrent.futures import Executor
from functools import partial

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

# Import GPU configuration before anything pulls in TensorFlow.
from face_util import gpu_utils as _gpu_utils  # noqa: F401
from face_util.anti_spoof_infer import (
    finalize_antispoof,
    get_antispoof_model,
    infer_antispoof_batch,
    prepare_antispoof,
)

from src.batching import MicroBatcher


# ============================================================
# WARMUP
# ============================================================

def warmup_antispoof() -> None:
    """
    Run on the GPU thread at startup: loads the models and executes a few
    batch sizes so CUDA kernels / cuDNN algorithms are selected before the
    first real session.
    """
    try:
        predictor = get_antispoof_model()

        for batch_size in (1, 8, 32):
            predictor.predict_batch(
                [
                    np.zeros(
                        (batch_size, m.h_input, m.w_input, 3),
                        dtype=np.uint8,
                    )
                    for m in predictor.models
                ]
            )

    except Exception as exc:
        print(f"[warmup] anti-spoof failed: {exc}")


def warmup_cpu_thread(face_analyzer=None) -> None:
    """Run on every CPU worker thread: per-thread face detector + landmarker."""
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)

    try:
        get_antispoof_model().get_bbox(dummy)
    except Exception as exc:
        print(f"[warmup] face detector failed: {exc}")

    if face_analyzer is not None:
        try:
            face_analyzer.warmup()
        except Exception as exc:
            print(f"[warmup] face landmarker failed: {exc}")


# ============================================================
# FRAME DECODING (runs in the CPU pool, never on the event loop)
# ============================================================

def decode_b64_image(data: str) -> bytes | None:
    if not data:
        return None

    data = data.strip()

    # Support: data:image/jpeg;base64,...
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]

    try:
        return base64.b64decode(data)
    except (ValueError, binascii.Error):
        return None


def decode_bgr(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None

    array = np.frombuffer(image_bytes, dtype=np.uint8)

    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def decode_frame(payload) -> tuple[np.ndarray | None, str | None]:
    """
    payload: raw JPEG/PNG bytes (binary websocket message) or a base64 string.
    Returns (image_bgr, None) or (None, error_code).
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        image_bytes = bytes(payload)
    elif isinstance(payload, str):
        image_bytes = decode_b64_image(payload)
    else:
        image_bytes = None

    if not image_bytes:
        return None, "bad_frame_encoding"

    image = decode_bgr(image_bytes)

    if image is None:
        return None, "cannot_decode_frame"

    return image, None


# ============================================================
# EXECUTOR HELPER
# ============================================================

async def run_in_pool(executor: Executor, func, *args, timeout: float):
    loop = asyncio.get_running_loop()

    return await asyncio.wait_for(
        loop.run_in_executor(executor, partial(func, *args)),
        timeout=timeout,
    )


# ============================================================
# LIVENESS
# ============================================================

async def process_liveness_frame(
    *,
    image: np.ndarray,
    cpu_executor: Executor,
    batcher: MicroBatcher,
    timeout: float,
) -> dict:
    """
    CPU part (bbox, heuristic, crops) in the CPU pool, then the model forward
    pass through the shared micro-batcher on the GPU thread.
    """
    started_at = time.perf_counter()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    item, early = await run_in_pool(
        cpu_executor,
        prepare_antispoof,
        image,
        timeout=timeout,
    )

    if early is not None:
        is_real, score, reason = early

    else:
        prediction = await asyncio.wait_for(
            batcher.submit(item),
            timeout=max(0.001, deadline - loop.time()),
        )

        is_real, score, reason = finalize_antispoof(item, prediction)

    return {
        "is_real": bool(is_real),
        "score": float(score),
        "reason": reason,
        "latency_ms": round(
            (time.perf_counter() - started_at) * 1000,
            2,
        ),
    }


# ============================================================
# WEBSOCKET HELPERS
# ============================================================

async def check_api_key(
    websocket: WebSocket,
    expected_api_key: str,
) -> bool:
    client_api_key = websocket.headers.get("x-api-key")

    if not client_api_key:
        await websocket.close(code=1008, reason="API key missing")
        return False

    if not secrets.compare_digest(client_api_key, expected_api_key):
        await websocket.close(code=1008, reason="Invalid API key")
        return False

    return True


async def receive_message(websocket: WebSocket, timeout: float) -> dict:
    """
    Accepts both protocols:
      - text:   {"type": "frame", "frame": "<base64>", "session_id": "..."}
      - binary: raw JPEG bytes  (treated as a frame; ~33% less bandwidth,
                no base64 decode)
    """
    raw = await asyncio.wait_for(websocket.receive(), timeout=timeout)

    if raw["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(raw.get("code", 1000))

    if raw.get("bytes") is not None:
        return {"type": "frame", "frame": raw["bytes"]}

    try:
        message = json.loads(raw.get("text") or "")
    except ValueError:
        return {"type": "invalid_json"}

    return message if isinstance(message, dict) else {"type": "invalid_json"}


def get_remaining_time(
    loop: asyncio.AbstractEventLoop,
    deadline: float,
) -> float:
    return max(0.0, deadline - loop.time())


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
            "latency_ms": result["latency_ms"],
            "elapsed_seconds": round(elapsed, 2),
            "remaining_seconds": round(remaining, 2),
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
            "elapsed_seconds": round(elapsed, 2),
            "decision": decision,
        }
    )