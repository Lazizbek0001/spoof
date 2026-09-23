from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import secrets
from concurrent.futures import Executor
from functools import partial

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image, ImageOps

from face_util import gpu_utils as _gpu_utils  # noqa: F401  (must load before TF)
from face_util.anti_spoof_infer import (
    finalize_antispoof,
    get_antispoof_model,
    prepare_antispoof,
)
from src.batching import MicroBatcher


# ---------------------------------------------------------------- warmup

def warmup_antispoof() -> None:
    """GPU thread: load models and run a few batch sizes so kernels are ready."""
    predictor = get_antispoof_model()
    for batch in (1, 8, 32):
        predictor.predict_batch(
            [np.zeros((batch, m.h_input, m.w_input, 3), np.uint8) for m in predictor.models]
        )


def warmup_cpu_thread(face_analyzer) -> None:
    """Every CPU worker thread: create its own face detector and landmarker."""
    get_antispoof_model().get_bbox(np.zeros((240, 320, 3), np.uint8))
    face_analyzer.warmup()


# ---------------------------------------------------------------- decoding

def decode_frame(payload) -> np.ndarray | None:
    """
    payload: raw JPEG bytes (binary websocket message) or a base64 string.
    Applies the EXIF orientation tag, which phones use instead of rotating pixels.
    """
    if isinstance(payload, str):
        data = payload.strip()
        if data.startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        try:
            payload = base64.b64decode(data)
        except (ValueError, binascii.Error):
            return None
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        return None

    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(payload)))
        return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- executors

async def run_in_pool(executor: Executor, func, *args, timeout: float):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(executor, partial(func, *args)), timeout=max(timeout, 0.001)
    )


async def antispoof(image: np.ndarray, cpu_pool: Executor, batcher: MicroBatcher, timeout: float) -> dict:
    """CPU prep in the pool, model forward through the shared GPU batcher."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    item, early = await run_in_pool(cpu_pool, prepare_antispoof, image, timeout=timeout)
    if early is None:
        prediction = await asyncio.wait_for(batcher.submit(item), timeout=max(deadline - loop.time(), 0.001))
        early = finalize_antispoof(item, prediction)

    is_real, score, reason = early
    return {"is_real": bool(is_real), "score": float(score), "reason": reason}


# ---------------------------------------------------------------- websocket

async def check_api_key(websocket: WebSocket, expected: str) -> bool:
    key = websocket.headers.get("x-api-key", "")
    if key and secrets.compare_digest(key, expected):
        return True
    await websocket.close(code=1008, reason="Invalid API key")
    return False


async def receive_message(websocket: WebSocket, timeout: float) -> dict:
    """
    text:   {"type": "frame", "frame": "<base64>", "session_id": "..."}
    binary: raw JPEG bytes (a frame; ~33% less bandwidth, no base64 decode)
    """
    raw = await asyncio.wait_for(websocket.receive(), timeout=max(timeout, 0.001))
    if raw["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(raw.get("code", 1000))
    if raw.get("bytes") is not None:
        return {"type": "frame", "frame": raw["bytes"]}
    try:
        message = json.loads(raw.get("text") or "")
    except ValueError:
        return {"type": "invalid_json"}
    return message if isinstance(message, dict) else {"type": "invalid_json"}