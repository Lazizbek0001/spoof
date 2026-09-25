from __future__ import annotations

import os

from session import Session

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # TF shares the GPU with torch

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from random import SystemRandom

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from face_util.anti_spoof_infer import infer_antispoof_batch
from face_util.gpu_utils import describe_runtime, get_runtime_info
from src.batching import MicroBatcher
from src.config import settings
from src.face_actions import FaceActionAnalyzer
from src.helpers import (
    antispoof,
    check_api_key,
    decode_frame,
    receive_message,
    run_in_pool,
    warmup_antispoof,
    warmup_cpu_thread,
)
from src.liveness import LivenessSession
from src.recognition import FaceRecognizer

ROOT = Path(__file__).resolve().parent.parent
LANDMARKER_MODEL = ROOT / "src" / "models" / "face_landmarker.task"
REFERENCE_IMAGE = ROOT / settings.reference_image

ACTIONS = ("turn_left", "turn_right", "close_left_eye", "close_right_eye", "smile")

STAGE_ANTISPOOF = "antispoof"   # liveness + identity, from the same frames
STAGE_ACTION = "action"         # one random challenge


# ---------------------------------------------------------------- lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    cv2.setNumThreads(1)  # parallelism comes from the pools, not OpenCV

    cpu_pool = ThreadPoolExecutor(settings.cpu_workers, thread_name_prefix="cpu")
    gpu_pool = ThreadPoolExecutor(1, thread_name_prefix="gpu")
    recog_pool = ThreadPoolExecutor(settings.recog_workers, thread_name_prefix="recog")

    state = app.state
    state.cpu_pool = cpu_pool
    state.recog_pool = recog_pool
    state.batcher = MicroBatcher(
        infer_antispoof_batch, gpu_pool,
        max_batch=settings.max_batch, max_wait=settings.max_batch_wait_ms / 1000,
    )
    state.face_analyzer = FaceActionAnalyzer(model_path=LANDMARKER_MODEL)
    state.recognizer = FaceRecognizer()
    state.active_sessions = 0

    print("[startup]\n" + describe_runtime())
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_pool, warmup_antispoof)
    await asyncio.gather(*(
        loop.run_in_executor(cpu_pool, warmup_cpu_thread, state.face_analyzer)
        for _ in range(settings.cpu_workers)
    ))
    await loop.run_in_executor(recog_pool, state.recognizer.warmup)
    if REFERENCE_IMAGE.is_file():
        await loop.run_in_executor(recog_pool, state.recognizer.reference_embedding, REFERENCE_IMAGE)
    state.batcher.start()
    print(f"[startup] ready cpu_workers={settings.cpu_workers} max_batch={settings.max_batch}")

    try:
        yield
    finally:
        await state.batcher.stop()
        state.face_analyzer.close()
        for pool in (cpu_pool, gpu_pool, recog_pool):
            pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Liveness API", lifespan=lifespan)

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    if not await check_api_key(websocket, settings.api_key):
        return
    state = websocket.app.state
    if state.active_sessions >= settings.max_sessions:
        await websocket.close(code=1013, reason="Server busy")
        return

    await websocket.accept()
    state.active_sessions += 1
    session = None
    try:
        if not REFERENCE_IMAGE.is_file():
            await websocket.send_json({"type": "error", "message": f"reference_image_not_found:{REFERENCE_IMAGE}"})
            return
        session = Session(websocket)
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if session:
            try:
                await session.fail(session.stage, "server_error", error=str(exc))
            except Exception:
                pass
    finally:
        state.active_sessions -= 1
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health():
    return {
        "ok": True,
        "runtime": get_runtime_info(),
        "active_sessions": app.state.active_sessions,
        "max_sessions": settings.max_sessions,
        "antispoof_batching": app.state.batcher.stats(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }