import asyncio
import base64
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from deepface import DeepFace

# Your anti-spoof + liveness modules
from face_util.anti_spoof_infer import predict_antispoof_score
from face_util.gpu_utils import describe_runtime, get_runtime_info


# ---------------------------------------------------------------------------
# Thread pool for CPU-bound work (anti-spoof, DeepFace, cv2)
# ---------------------------------------------------------------------------
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="liveness")


# ---------------------------------------------------------------------------
# Lifespan: warm up models so the first request is fast
# ---------------------------------------------------------------------------
def _warmup_antispoof() -> None:
    try:
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        predict_antispoof_score(dummy)  # triggers model load, returns no_face_bbox
    except Exception as e:
        print(f"[warmup] antispoof warmup failed: {e}")


def _warmup_deepface() -> None:
    try:
        DeepFace.represent(
            img_path=np.zeros((160, 160, 3), dtype=np.uint8),
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False,
        )
    except Exception as e:
        print(f"[warmup] deepface warmup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup]", describe_runtime())

    loop = asyncio.get_running_loop()
    # Warm up in parallel
    await asyncio.gather(
        loop.run_in_executor(EXECUTOR, _warmup_antispoof),
        loop.run_in_executor(EXECUTOR, _warmup_deepface),
    )
    print("[startup] warmup done")

    yield

    print("[shutdown] closing executor")
    EXECUTOR.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

@app.get("/debug/routes")
async def debug_routes():
    return [
        {
            "path": r.path,
            "methods": list(getattr(r, "methods", [])),
            "name": r.name,
            "type": type(r).__name__,
        }
        for r in app.routes
    ]
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _decode_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _decode_b64_to_bytes(data: str) -> Optional[bytes]:
    """
    Accepts either:
      - raw base64 string
      - data URI: "data:image/jpeg;base64,....."
    """
    if not data:
        return None
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sliding-window liveness engine (per session)
# ---------------------------------------------------------------------------
class LivenessSession:
    def __init__(
        self,
        min_frames: int = 8,
        min_real_frames: int = 6,
        min_avg_score: float = 0.90,
        required_real_ratio: float = 0.75,
    ):
        self.min_frames = min_frames
        self.min_real_frames = min_real_frames
        self.min_avg_score = min_avg_score
        self.required_real_ratio = required_real_ratio

        self.frames = []
        self.created_at = time.time()

    def add(
        self,
        is_real: bool,
        score: float,
        reason: str,
        frame_index: int,
    ) -> dict:

        self.frames.append(
            {
                "index": frame_index,
                "is_real": is_real,
                "score": float(score),
                "reason": reason,
                "ts": time.time(),
            }
        )

        return self._decision()

    def _decision(self) -> dict:
        n = len(self.frames)

        if n == 0:
            return {
                "ready": False,
                "ok": False,
                "reason": "no_frames",
            }

        real_count = sum(
            1
            for f in self.frames
            if f["is_real"]
        )

        fake_count = n - real_count
        real_ratio = real_count / n

        real_scores = [
            f["score"]
            for f in self.frames
            if f["is_real"]
        ]

        avg_score = (
            float(sum(real_scores) / len(real_scores))
            if real_scores
            else 0.0
        )

        best = max(
            self.frames,
            key=lambda f: f["score"],
        )

        # Hard reject screen replay
        if any(
            f["reason"].startswith("screen_")
            for f in self.frames
        ):
            return {
                "ready": True,
                "ok": False,
                "reason": "screen_replay_detected",
                "total_frames": n,
                "real_count": real_count,
                "fake_count": fake_count,
                "real_ratio": round(real_ratio, 3),
                "avg_score": round(avg_score, 4),
                "best_frame_index": best["index"],
            }

        # Collect minimum amount first
        if n < self.min_frames:
            return {
                "ready": False,
                "ok": False,
                "reason": f"collecting:{n}/{self.min_frames}",
                "total_frames": n,
                "real_count": real_count,
                "fake_count": fake_count,
                "real_ratio": round(real_ratio, 3),
                "avg_score": round(avg_score, 4),
                "best_frame_index": best["index"],
            }

        if real_count < self.min_real_frames:
            return {
                "ready": True,
                "ok": False,
                "reason": "not_enough_real_frames",
                "total_frames": n,
                "real_count": real_count,
                "fake_count": fake_count,
                "real_ratio": round(real_ratio, 3),
                "avg_score": round(avg_score, 4),
                "best_frame_index": best["index"],
            }

        if real_ratio < self.required_real_ratio:
            return {
                "ready": True,
                "ok": False,
                "reason": "real_ratio_too_low",
                "total_frames": n,
                "real_count": real_count,
                "fake_count": fake_count,
                "real_ratio": round(real_ratio, 3),
                "avg_score": round(avg_score, 4),
                "best_frame_index": best["index"],
            }

        if avg_score < self.min_avg_score:
            return {
                "ready": True,
                "ok": False,
                "reason": "avg_score_too_low",
                "total_frames": n,
                "real_count": real_count,
                "fake_count": fake_count,
                "real_ratio": round(real_ratio, 3),
                "avg_score": round(avg_score, 4),
                "best_frame_index": best["index"],
            }

        return {
            "ready": True,
            "ok": True,
            "reason": "ok",
            "total_frames": n,
            "real_count": real_count,
            "fake_count": fake_count,
            "real_ratio": round(real_ratio, 3),
            "avg_score": round(avg_score, 4),
            "best_frame_index": best["index"],
        }
# In-memory session registry (per device / connection)
SESSIONS: Dict[str, LivenessSession] = {}


def _run_liveness_on_frame(image_bytes: bytes) -> dict:
    """
    Runs the anti-spoof model on a single frame. Blocking — call via executor.
    """
    img = _decode_bgr(image_bytes)
    if img is None:
        return {"is_real": False, "score": 0.0, "reason": "decode_failed"}

    t0 = time.time()
    is_real, score, reason = predict_antispoof_score(img)
    dt = time.time() - t0

    return {
        "is_real": bool(is_real),
        "score": float(score),
        "reason": reason,
        "latency_ms": round(dt * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Optional: static API key check (skip if you already have auth)
# ---------------------------------------------------------------------------
API_KEY = "CHANGE_ME"  # override via env in production


def _check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and API_KEY != "CHANGE_ME" and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid_api_key")


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    WebSocket streaming endpoint.

    Protocol (JSON messages, both directions):

      Client -> Server:
        {
          "type": "frame",
          "session_id": "abc-123",       # any string, device-generated
          "frame": "<base64 or data URI>",
          "frame_index": 42              # optional, for debugging
        }
        {
          "type": "reset"
        }
        {
          "type": "close"
        }

      Server -> Client:
        {
          "type": "ack",
          "frame_index": 42,
          "is_real": true,
          "score": 0.93,
          "reason": "label=0,real_score=0.9300",
          "latency_ms": 47.2,
          "decision": { ...LivenessSession decision... }
        }
        {
          "type": "error",
          "message": "..."
        }

    The client is expected to send frames at ~5–15 fps.
    Server throttles: it processes every frame but caps in-flight work.
    """
    await websocket.accept()

    session_id: Optional[str] = None
    session: Optional[LivenessSession] = None
    frame_counter = 0
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await websocket.receive_json()

            mtype = msg.get("type")

            if mtype == "reset":
                session_id = None
                session = None
                frame_counter = 0
                await websocket.send_json({"type": "reset_ok"})
                continue

            if mtype == "close":
                await websocket.close()
                return

            if mtype != "frame":
                await websocket.send_json(
                    {"type": "error", "message": f"unknown_type:{mtype}"}
                )
                continue

            # ---- resolve session ----
            sid = str(msg.get("session_id") or "default")
            if session is None or sid != session_id:
                session_id = sid
                session = SESSIONS.setdefault(
                    session_id,
                    LivenessSession(
                        min_frames=12,
                        min_real_frames=8,
                        min_avg_score=0.90,
                        required_real_ratio=0.80,
                    ),
                )

            # ---- decode frame ----
            raw_b64 = msg.get("frame")
            image_bytes = _decode_b64_to_bytes(raw_b64 or "")
            if not image_bytes:
                await websocket.send_json(
                    {"type": "error", "message": "bad_frame_encoding"}
                )
                continue

            frame_counter += 1

            # ---- run liveness in thread pool ----
            try:
                result = await loop.run_in_executor(
                    EXECUTOR, _run_liveness_on_frame, image_bytes
                )
            except Exception as e:
                await websocket.send_json(
                    {"type": "error", "message": f"liveness_error:{e}"}
                )
                continue

            # ---- feed session window ----
            decision = session.add(
                is_real=result["is_real"],
                score=result["score"],
                reason=result["reason"],
                frame_index=frame_counter,
            )

            await websocket.send_json(
                {
                    "type": "ack",
                    "frame_index": frame_counter,
                    "is_real": result["is_real"],
                    "score": result["score"],
                    "reason": result["reason"],
                    "latency_ms": result["latency_ms"],
                    "decision": decision,
                }
            )

            # Optional: auto-close the session when a decision is reached
            if decision.get("ready") and decision.get("ok"):
                await websocket.send_json(
                    {"type": "final", "ok": True, "decision": decision}
                )
                # Don't close the socket — let client decide

    except WebSocketDisconnect:
        # normal client disconnect
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": f"server_error:{e}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP fallback: single-shot liveness on a frame
# ---------------------------------------------------------------------------
class FrameRequest(BaseModel):
    session_id: str
    frame: str          # base64 or data URI
    frame_index: Optional[int] = None


@app.post("/stream/frame")
async def stream_frame(
    body: FrameRequest,
    _: None = Depends(_check_api_key),
):
    """
    HTTP fallback for devices that cannot use WebSockets.
    Send one frame per request. Session state is kept in memory by session_id.
    """
    image_bytes = _decode_b64_to_bytes(body.frame)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="bad_frame_encoding")

    session = SESSIONS.setdefault(
        body.session_id,
        LivenessSession(
            window_size=8,
            min_real_frames=2,
            min_avg_score=0.60,
            required_real_ratio=0.30,
        ),
    )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(EXECUTOR, _run_liveness_on_frame, image_bytes)

    idx = body.frame_index if body.frame_index is not None else len(session.frames) + 1
    decision = session.add(
        is_real=result["is_real"],
        score=result["score"],
        reason=result["reason"],
        frame_index=idx,
    )

    return {
        "session_id": body.session_id,
        "frame_index": idx,
        "is_real": result["is_real"],
        "score": result["score"],
        "reason": result["reason"],
        "latency_ms": result["latency_ms"],
        "decision": decision,
    }


@app.delete("/stream/session/{session_id}")
async def reset_session(
    session_id: str,
    _: None = Depends(_check_api_key),
):
    SESSIONS.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@app.get("/stream/sessions")
async def list_sessions(_: None = Depends(_check_api_key)):
    now = time.time()
    return {
        "count": len(SESSIONS),
        "sessions": [
            {
                "session_id": sid,
                "frames": len(s.frames),
                "age_s": round(now - s.created_at, 1),
            }
            for sid, s in SESSIONS.items()
        ],
    }


# ---------------------------------------------------------------------------
# Optional: existing image upload endpoint (kept for backward compat)
# ---------------------------------------------------------------------------
@app.post("/verify/liveness")
async def verify_liveness(
    file: UploadFile = File(...),
    _: None = Depends(_check_api_key),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty_file")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(EXECUTOR, _run_liveness_on_frame, image_bytes)
    return result


@app.get("/health")
async def health():
    return {
        "ok": True,
        "runtime": get_runtime_info(),
        "ts": datetime.now(timezone.utc).isoformat(),
    }