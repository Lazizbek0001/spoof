from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # TF shares the GPU with torch

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from face_util.anti_spoof_infer import infer_antispoof_batch
from face_util.gpu_utils import describe_runtime, get_runtime_info
from src.batching import MicroBatcher
from src.config import settings
from src.face_actions import FaceActionAnalyzer
from src.helpers import check_api_key, decode_frame, run_in_pool, warmup_antispoof, warmup_cpu_thread
from src.recognition import FaceRecognizer
from src.session import Session

LANDMARKER_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


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


# ---------------------------------------------------------------- websocket

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


# ---------------------------------------------------------------- enrollment

class EmbedRequest(BaseModel):
    image: str   # base64 JPEG/PNG (data: URL prefix allowed)


@app.post("/embed")
async def embed(body: EmbedRequest, x_api_key: str = Header(default="")):
    """
    Turn an enrollment photo into the reference vector the client later sends
    over the websocket. Uses the same model as the stream, so vectors match.
    """
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid API key")

    state = app.state
    frame = await run_in_pool(state.cpu_pool, decode_frame, body.image, timeout=10)
    if frame is None:
        raise HTTPException(400, "cannot_decode_image")
    try:
        vector = await run_in_pool(state.recog_pool, state.recognizer.embed, frame, timeout=30)
    except ValueError:
        raise HTTPException(422, "face_not_detected")

    return {"model": state.recognizer.MODEL, "dim": state.recognizer.DIM, "embedding": vector.tolist()}



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