from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TensorFlow shares the GPU with PyTorch: never let it grab all VRAM.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import asyncio
import threading
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
    check_api_key,
    decode_frame,
    get_remaining_time,
    process_liveness_frame,
    receive_message,
    run_in_pool,
    send_ack,
    send_final,
    warmup_antispoof,
    warmup_cpu_thread,
)
from src.liveness import LivenessSession
from src.recognition import FaceRecognizer


# ============================================================
# CONFIG
# ============================================================

# Paths are resolved against the project, not the current working directory.
_SRC_DIR = Path(__file__).resolve().parent

FACE_LANDMARKER_MODEL = _SRC_DIR / "models" / "face_landmarker.task"

REFERENCE_IMAGE = Path(settings.reference_image)

if not REFERENCE_IMAGE.is_absolute():
    REFERENCE_IMAGE = _SRC_DIR.parent / REFERENCE_IMAGE

FACE_MATCHES_REQUIRED = settings.face_matches_required

ACTION_MATCHES_REQUIRED = settings.action_matches_required

FACE_ACTIONS = (
    "turn_left",
    "turn_right",
    "close_left_eye",
    "close_right_eye",
    "smile",
)

_random = SystemRandom()


# ============================================================
# VERIFICATION STATE
# ============================================================

class VerificationStage:
    ANTISPOOF = "antispoof"
    FACE_RECOGNITION = "face_recognition"
    ACTION = "action"
    DONE = "done"


class VerificationSession:
    def __init__(self, reference_image: str | Path):
        self.stage = VerificationStage.ANTISPOOF
        self.reference_image = str(reference_image)

        # Anti-spoof
        self.liveness_decision: dict | None = None

        # Face recognition
        self.identity_verified = False
        self.face_match_count = 0

        # Actions: random order makes replaying a predefined sequence harder.
        self.actions = _random.sample(list(FACE_ACTIONS), len(FACE_ACTIONS))
        self.action_index = 0
        self.action_match_count = 0
        self.completed_actions: list[str] = []

    @property
    def current_action(self) -> str | None:
        if self.action_index >= len(self.actions):
            return None
        return self.actions[self.action_index]

    def complete_current_action(self) -> str | None:
        action = self.current_action
        if action is None:
            return None

        self.completed_actions.append(action)
        self.action_index += 1
        self.action_match_count = 0
        return action


def create_liveness_session() -> LivenessSession:
    return LivenessSession(
        window_size=settings.window_size,
        min_real_frames=settings.min_real_frames,
        min_avg_score=settings.min_avg_score,
        required_real_ratio=settings.required_real_ratio,
    )


# ============================================================
# LIFESPAN
# ============================================================

async def _warm_cpu_threads(
    pool: ThreadPoolExecutor,
    count: int,
    analyzer: FaceActionAnalyzer,
) -> None:
    """
    Force the pool to spawn all its threads and initialise the per-thread
    face detector + landmarker in each, so no session pays that cost.
    The barrier keeps every task busy until all threads have started.
    """
    barrier = threading.Barrier(count)

    def task() -> None:
        warmup_cpu_thread(analyzer)
        try:
            barrier.wait(timeout=30)
        except threading.BrokenBarrierError:
            pass

    loop = asyncio.get_running_loop()

    await asyncio.gather(
        *(loop.run_in_executor(pool, task) for _ in range(count))
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Parallelism comes from the pools; OpenCV's own threads would only
    # oversubscribe the cores.
    cv2.setNumThreads(1)

    # CPU pool : decode, face bbox, screen heuristic, crops, MediaPipe
    # GPU pool : exactly one thread, fed with batches by the micro-batcher
    # Recog    : DeepFace / TensorFlow
    cpu_pool = ThreadPoolExecutor(settings.cpu_workers, thread_name_prefix="cpu")
    gpu_pool = ThreadPoolExecutor(1, thread_name_prefix="gpu")
    recog_pool = ThreadPoolExecutor(settings.recog_workers, thread_name_prefix="recog")

    batcher = MicroBatcher(
        infer_antispoof_batch,
        gpu_pool,
        max_batch=settings.max_batch,
        max_wait=settings.max_batch_wait_ms / 1000.0,
    )

    if not FACE_LANDMARKER_MODEL.is_file():
        raise RuntimeError(
            f"Face Landmarker model not found: {FACE_LANDMARKER_MODEL}"
        )

    face_analyzer = FaceActionAnalyzer(model_path=FACE_LANDMARKER_MODEL)
    recognizer = FaceRecognizer()

    app.state.cpu_pool = cpu_pool
    app.state.recog_pool = recog_pool
    app.state.batcher = batcher
    app.state.face_analyzer = face_analyzer
    app.state.recognizer = recognizer
    app.state.active_sessions = 0

    print("[startup]\n" + describe_runtime())

    loop = asyncio.get_running_loop()

    await loop.run_in_executor(gpu_pool, warmup_antispoof)
    await _warm_cpu_threads(cpu_pool, settings.cpu_workers, face_analyzer)
    await loop.run_in_executor(recog_pool, recognizer.warmup)

    if REFERENCE_IMAGE.is_file():
        try:
            await loop.run_in_executor(
                recog_pool,
                recognizer.reference_embedding,
                REFERENCE_IMAGE,
            )
        except Exception as exc:
            print(f"[startup] reference embedding failed: {exc}")

    batcher.start()

    print(
        "[startup] warmup done "
        f"(cpu_workers={settings.cpu_workers}, "
        f"recog_workers={settings.recog_workers}, "
        f"max_batch={settings.max_batch}, "
        f"max_sessions={settings.max_sessions})"
    )

    try:
        yield

    finally:
        print("[shutdown] closing models")

        await batcher.stop()

        try:
            face_analyzer.close()
        except Exception:
            pass

        for pool in (cpu_pool, gpu_pool, recog_pool):
            pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Liveness API",
    lifespan=lifespan,
)


# ============================================================
# ONE WEBSOCKET CONNECTION
# ============================================================

class StreamHandler:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.state = websocket.app.state

        self.loop = asyncio.get_running_loop()
        self.started_at = self.loop.time()
        self.deadline = self.started_at + settings.max_connection_seconds

        self.session_id: str | None = None
        self.challenge_results: dict[str, dict] = {}
        self._reset_session()

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    def _reset_session(self) -> None:
        self.verification = VerificationSession(REFERENCE_IMAGE)
        self.liveness: LivenessSession | None = None
        self.frame_counter = 0
        self.challenge_results.clear()

    @property
    def remaining(self) -> float:
        return get_remaining_time(self.loop, self.deadline)

    @property
    def elapsed(self) -> float:
        return self.loop.time() - self.started_at

    def current_challenge(self) -> str:
        stage = self.verification.stage

        if stage == VerificationStage.ACTION:
            return self.verification.current_action or "action"

        if stage in (
            VerificationStage.ANTISPOOF,
            VerificationStage.FACE_RECOGNITION,
        ):
            return stage

        return "unknown"

    def save_result(
        self,
        challenge: str,
        passed: bool,
        reason: str,
        details: dict | None = None,
    ) -> None:
        self.challenge_results[challenge] = {
            "challenge": challenge,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "reason": reason,
            "details": details or {},
        }

    # --------------------------------------------------------
    # Final messages
    # --------------------------------------------------------

    async def finish(self, *, ok: bool, reason: str) -> None:
        await send_final(
            self.ws,
            ok=ok,
            reason=reason,
            decision={
                "liveness": self.verification.liveness_decision is not None,
                "face_verified": self.verification.identity_verified,
                "completed_actions": self.verification.completed_actions,
                "challenge_results": self.challenge_results,
            },
            elapsed=self.elapsed,
        )

    async def fail(
        self,
        challenge: str,
        reason: str,
        details: dict | None = None,
    ) -> None:
        self.save_result(challenge, False, reason, details)
        await self.finish(ok=False, reason=reason)

    async def handle_timeout(self) -> None:
        challenge = self.current_challenge()
        existing = self.challenge_results.get(challenge)

        # Do not replace an already-passed challenge.
        if not (existing and existing.get("passed", False)):
            self.save_result(
                challenge,
                False,
                "timeout",
                {"max_seconds": settings.max_connection_seconds},
            )

        await self.finish(ok=False, reason="connection_timeout")

    # --------------------------------------------------------
    # Model calls
    # --------------------------------------------------------

    async def _liveness(self, frame) -> dict:
        return await process_liveness_frame(
            image=frame,
            cpu_executor=self.state.cpu_pool,
            batcher=self.state.batcher,
            timeout=self.remaining,
        )

    async def _verify(self, frame) -> dict:
        return await run_in_pool(
            self.state.recog_pool,
            self.state.recognizer.verify,
            self.verification.reference_image,
            frame,
            timeout=self.remaining,
        )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    async def run(self) -> None:
        try:
            while True:
                if self.remaining <= 0:
                    await self.handle_timeout()
                    return

                message = await receive_message(self.ws, self.remaining)
                message_type = message.get("type")

                if message_type == "close":
                    return

                if message_type == "reset":
                    self.session_id = None
                    self._reset_session()

                    await self.ws.send_json(
                        {
                            "type": "reset_ok",
                            "stage": VerificationStage.ANTISPOOF,
                            "remaining_seconds": round(self.remaining, 2),
                        }
                    )
                    continue

                if message_type != "frame":
                    await self.ws.send_json(
                        {
                            "type": "error",
                            "message": f"unknown_type:{message_type}",
                        }
                    )
                    continue

                # Binary frames carry no session id: keep the current one.
                incoming = str(
                    message.get("session_id")
                    or (
                        self.session_id
                        if isinstance(message.get("frame"), bytes)
                        else None
                    )
                    or "default"
                )

                if self.liveness is None or incoming != self.session_id:
                    self.session_id = incoming
                    self._reset_session()
                    self.liveness = create_liveness_session()

                # JPEG / base64 decoding happens in the CPU pool.
                frame, error = await run_in_pool(
                    self.state.cpu_pool,
                    decode_frame,
                    message.get("frame", ""),
                    timeout=max(self.remaining, 0.001),
                )

                if error:
                    await self.ws.send_json({"type": "error", "message": error})
                    continue

                self.frame_counter += 1

                if self.remaining <= 0:
                    await self.handle_timeout()
                    return

                stage = self.verification.stage

                if stage == VerificationStage.ANTISPOOF:
                    done = await self.stage_antispoof(frame)
                elif stage == VerificationStage.FACE_RECOGNITION:
                    done = await self.stage_recognition(frame)
                elif stage == VerificationStage.ACTION:
                    done = await self.stage_action(frame)
                else:
                    done = True

                if done:
                    return

        except asyncio.TimeoutError:
            await self.handle_timeout()

    # --------------------------------------------------------
    # STAGE 1 — ANTI-SPOOF
    # --------------------------------------------------------

    async def stage_antispoof(self, frame) -> bool:
        try:
            result = await self._liveness(frame)

        except asyncio.TimeoutError:
            raise

        except Exception as exc:
            await self.fail("antispoof", "liveness_error", {"error": str(exc)})
            return True

        decision = self.liveness.add(
            is_real=result["is_real"],
            score=result["score"],
            reason=result["reason"],
            frame_index=self.frame_counter,
        )

        await send_ack(
            self.ws,
            frame_index=self.frame_counter,
            result=result,
            decision=decision,
            elapsed=self.elapsed,
            remaining=self.remaining,
        )

        if decision.get("reason") == "screen_replay_detected":
            await self.fail(
                "antispoof",
                "screen_replay_detected",
                {"decision": decision},
            )
            return True

        if decision.get("ready") and decision.get("ok"):
            self.verification.liveness_decision = decision

            self.save_result(
                "antispoof",
                True,
                "liveness_passed",
                {
                    key: decision.get(key)
                    for key in (
                        "real_count",
                        "fake_count",
                        "real_ratio",
                        "avg_score",
                    )
                },
            )

            self.verification.stage = VerificationStage.FACE_RECOGNITION

            await self.ws.send_json(
                {
                    "type": "liveness_passed",
                    "ok": True,
                    "next_stage": "face_recognition",
                }
            )

        return False

    # --------------------------------------------------------
    # STAGE 2 — FACE RECOGNITION
    # --------------------------------------------------------

    async def _send_recognition_miss(self, reason: str) -> None:
        self.verification.face_match_count = 0

        await self.ws.send_json(
            {
                "type": "face_recognition",
                "verified": False,
                "identity_verified": False,
                "reason": reason,
                "match_count": 0,
                "required_matches": FACE_MATCHES_REQUIRED,
            }
        )

    async def stage_recognition(self, frame) -> bool:
        session = self.verification

        try:
            if settings.recheck_liveness_on_recognition:
                # Same frame goes through anti-spoof and recognition in
                # parallel (GPU batcher + TF), so this costs no extra latency.
                live, verification = await asyncio.gather(
                    self._liveness(frame),
                    self._verify(frame),
                    return_exceptions=True,
                )

                for outcome in (verification, live):
                    if isinstance(outcome, asyncio.TimeoutError):
                        raise outcome

                if isinstance(verification, Exception):
                    raise verification

                if isinstance(live, Exception):
                    raise live

                if not live["is_real"]:
                    await self._send_recognition_miss("spoof_suspected")
                    return False

            else:
                verification = await self._verify(frame)

        except asyncio.TimeoutError:
            raise

        except ValueError:
            await self._send_recognition_miss("face_not_detected")
            return False

        except Exception as exc:
            await self.fail(
                "face_recognition",
                "face_recognition_error",
                {"error": str(exc)},
            )
            return True

        # Require consecutive matches.
        if verification["verified"]:
            session.face_match_count += 1
        else:
            session.face_match_count = 0

        if session.face_match_count >= FACE_MATCHES_REQUIRED:
            session.identity_verified = True
            session.stage = VerificationStage.ACTION

        response = {
            "type": "face_recognition",
            "verified": verification.get("verified", False),
            "identity_verified": session.identity_verified,
            "match_count": session.face_match_count,
            "required_matches": FACE_MATCHES_REQUIRED,
            "distance": verification.get("distance"),
            "threshold": verification.get("threshold"),
        }

        if session.identity_verified:
            self.save_result(
                "face_recognition",
                True,
                "face_matched",
                {
                    "match_count": session.face_match_count,
                    "required_matches": FACE_MATCHES_REQUIRED,
                    "distance": verification.get("distance"),
                    "threshold": verification.get("threshold"),
                },
            )

            response["next_stage"] = "action"
            response["action"] = session.current_action

        await self.ws.send_json(response)

        return False

    # --------------------------------------------------------
    # STAGE 3 — ONE ACTION ONLY
    # --------------------------------------------------------

    async def stage_action(self, frame) -> bool:
        session = self.verification
        action = session.current_action

        if action is None:
            await self.fail("action", "no_action_available")
            return True

        try:
            # Per-thread landmarkers: no global lock, sessions run in parallel.
            result = await run_in_pool(
                self.state.cpu_pool,
                self.state.face_analyzer.check_action,
                frame,
                action,
                timeout=self.remaining,
            )

        except asyncio.TimeoutError:
            raise

        except Exception as exc:
            await self.fail(action, "face_action_error", {"error": str(exc)})
            return True

        # Require consecutive detections.
        if result["ok"]:
            session.action_match_count += 1
        else:
            session.action_match_count = 0

        detected = bool(result.get("ok", False))
        reason = result.get("reason")

        passed = (
            detected
            and session.action_match_count >= ACTION_MATCHES_REQUIRED
        )

        # The person performing the action must still be the recognised one.
        if passed and settings.verify_identity_on_action:
            try:
                identity = await self._verify(frame)
                same_person = bool(identity["verified"])

            except asyncio.TimeoutError:
                raise

            except Exception:
                same_person = False

            if not same_person:
                passed = False
                detected = False
                reason = "identity_mismatch"
                session.action_match_count = 0

        matched_count = session.action_match_count

        response = {
            "type": "action_result",
            "action": action,
            "detected": detected,
            "reason": reason,
            "match_count": matched_count,
            "required_matches": ACTION_MATCHES_REQUIRED,
            "action_completed": passed,
            "completed_action": None,
            "completed_actions": session.completed_actions,
            "next_action": action,
            "analysis": result.get("analysis"),
        }

        if not passed:
            await self.ws.send_json(response)
            return False

        completed = session.complete_current_action() or action

        self.save_result(
            completed,
            True,
            "action_detected",
            {
                "matches": matched_count,
                "required_matches": ACTION_MATCHES_REQUIRED,
                "analysis": result.get("analysis"),
            },
        )

        response["completed_action"] = completed
        response["completed_actions"] = session.completed_actions
        response["next_action"] = None

        await self.ws.send_json(response)

        # One action = done immediately.
        session.stage = VerificationStage.DONE

        await self.finish(ok=True, reason="verification_complete")

        return True



@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    if not await check_api_key(websocket, settings.api_key):
        return

    state = websocket.app.state

    # Admission control: refuse instead of degrading every active session.
    if state.active_sessions >= settings.max_sessions:
        await websocket.close(code=1013, reason="Server busy")
        return

    await websocket.accept()

    state.active_sessions += 1

    handler: StreamHandler | None = None

    try:
        if not REFERENCE_IMAGE.is_file():
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"reference_image_not_found:{REFERENCE_IMAGE}",
                }
            )
            return

        handler = StreamHandler(websocket)

        await handler.run()

    except WebSocketDisconnect:
        pass

    except Exception as exc:
        try:
            if handler is not None:
                await handler.fail(
                    handler.current_challenge(),
                    "server_error",
                    {"error": str(exc)},
                )
        except Exception:
            pass

    finally:
        state.active_sessions -= 1

        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================
# HEALTH
# ============================================================

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