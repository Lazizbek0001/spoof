import asyncio
import base64
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dotenv import load_dotenv
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
load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is not configured")

def _check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and API_KEY != "CHANGE_ME" and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid_api_key")


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    MAX_CONNECTION_SECONDS = 30.0
    client_api_key = websocket.headers.get("x-api-key")

    if (
        not client_api_key
        or not secrets.compare_digest(client_api_key, API_KEY)
    ):
        await websocket.close(
            code=1008,
            reason="Invalid API key",
        )
        return

    await websocket.accept()

    session_id: Optional[str] = None
    session: Optional[LivenessSession] = None
    frame_counter = 0

    loop = asyncio.get_running_loop()

    # 30-second timer starts when WebSocket is accepted
    started_at = loop.time()
    deadline = started_at + MAX_CONNECTION_SECONDS

    try:
        while True:
            # ---------------------------------------------------------
            # Calculate remaining connection time
            # ---------------------------------------------------------
            remaining = deadline - loop.time()

            if remaining <= 0:
                await websocket.send_json(
                    {
                        "type": "final",
                        "ok": False,
                        "reason": "connection_timeout",
                        "message": "Liveness session exceeded 30 seconds.",
                    }
                )
                break

            # ---------------------------------------------------------
            # Wait for message, but never longer than remaining time
            # ---------------------------------------------------------
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=remaining,
                )

            except asyncio.TimeoutError:
                await websocket.send_json(
                    {
                        "type": "final",
                        "ok": False,
                        "reason": "connection_timeout",
                        "message": "Liveness session exceeded 30 seconds.",
                    }
                )
                break

            mtype = msg.get("type")

            # ---------------------------------------------------------
            # RESET
            # ---------------------------------------------------------
            if mtype == "reset":
                if session_id:
                    SESSIONS.pop(session_id, None)

                session_id = None
                session = None
                frame_counter = 0

                # IMPORTANT:
                # Timer is NOT restarted here.
                # Maximum connection lifetime remains 30 seconds
                # from websocket.accept().

                await websocket.send_json(
                    {
                        "type": "reset_ok",
                        "remaining_seconds": round(
                            max(0.0, deadline - loop.time()),
                            2,
                        ),
                    }
                )
                continue

            # ---------------------------------------------------------
            # CLOSE
            # ---------------------------------------------------------
            if mtype == "close":
                break

            # ---------------------------------------------------------
            # INVALID MESSAGE
            # ---------------------------------------------------------
            if mtype != "frame":
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"unknown_type:{mtype}",
                    }
                )
                continue

            # ---------------------------------------------------------
            # Resolve session
            # ---------------------------------------------------------
            sid = str(
                msg.get("session_id") or "default"
            )

            if session is None or sid != session_id:
                # Remove previous session if client changes session_id
                if session_id and session_id != sid:
                    SESSIONS.pop(session_id, None)

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

            # ---------------------------------------------------------
            # Decode frame
            # ---------------------------------------------------------
            raw_b64 = msg.get("frame")

            image_bytes = _decode_b64_to_bytes(
                raw_b64 or ""
            )

            if not image_bytes:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "bad_frame_encoding",
                    }
                )
                continue

            frame_counter += 1

            # ---------------------------------------------------------
            # Check timeout again before inference
            # ---------------------------------------------------------
            remaining = deadline - loop.time()

            if remaining <= 0:
                await websocket.send_json(
                    {
                        "type": "final",
                        "ok": False,
                        "reason": "connection_timeout",
                        "message": "Liveness session exceeded 30 seconds.",
                    }
                )
                break

            # ---------------------------------------------------------
            # Run liveness
            # ---------------------------------------------------------
            try:
                future = loop.run_in_executor(
                    EXECUTOR,
                    _run_liveness_on_frame,
                    image_bytes,
                )

                # Inference is also bounded by the 30-second deadline
                result = await asyncio.wait_for(
                    future,
                    timeout=remaining,
                )

            except asyncio.TimeoutError:
                await websocket.send_json(
                    {
                        "type": "final",
                        "ok": False,
                        "reason": "connection_timeout",
                        "message": "Liveness session exceeded 30 seconds.",
                    }
                )
                break

            except Exception as e:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"liveness_error:{e}",
                    }
                )
                continue

            # ---------------------------------------------------------
            # Add result to session
            # ---------------------------------------------------------
            decision = session.add(
                is_real=result["is_real"],
                score=result["score"],
                reason=result["reason"],
                frame_index=frame_counter,
            )

            elapsed = loop.time() - started_at

            remaining_seconds = max(
                0.0,
                deadline - loop.time(),
            )

            # ---------------------------------------------------------
            # ACK
            # ---------------------------------------------------------
            await websocket.send_json(
                {
                    "type": "ack",
                    "frame_index": frame_counter,
                    "is_real": result["is_real"],
                    "score": result["score"],
                    "reason": result["reason"],
                    "latency_ms": result["latency_ms"],
                    "elapsed_seconds": round(elapsed, 2),
                    "remaining_seconds": round(
                        remaining_seconds,
                        2,
                    ),
                    "decision": decision,
                }
            )

            # ---------------------------------------------------------
            # PASS
            # ---------------------------------------------------------
            if (
                decision.get("ready")
                and decision.get("ok")
            ):
                await websocket.send_json(
                    {
                        "type": "final",
                        "ok": True,
                        "reason": "liveness_passed",
                        "elapsed_seconds": round(
                            elapsed,
                            2,
                        ),
                        "decision": decision,
                    }
                )

                # Authentication attempt is complete.
                break

    except WebSocketDisconnect:
        pass

    except Exception as e:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"server_error:{e}",
                }
            )
        except Exception:
            pass

    finally:
        # -------------------------------------------------------------
        # Always remove session so old frames cannot affect next attempt
        # -------------------------------------------------------------
        if session_id:
            SESSIONS.pop(session_id, None)

        try:
            await websocket.close()
        except Exception:
            pass



@app.get("/health")
async def health():
    return {
        "ok": True,
        "runtime": get_runtime_info(),
        "ts": datetime.now(timezone.utc).isoformat(),
    }