from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from random import SystemRandom

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)

from src.face_actions import FaceActionAnalyzer
from face_util.gpu_utils import (
    describe_runtime,
    get_runtime_info,
)

from src.config import settings
from src.helpers import (
    check_api_key,
    decode_b64_image,
    decode_bgr,
    get_remaining_time,
    process_liveness_frame,
    send_ack,
    send_final,
    send_timeout,
    warmup_antispoof,
    warmup_deepface,
)
from src.liveness import LivenessSession


# ============================================================
# CONFIG
# ============================================================

FACE_LANDMARKER_MODEL = Path(
    "src/models/face_landmarker.task"
)

REFERENCE_IMAGE = Path(
    "sample/5.jpg"
)

FACE_MATCHES_REQUIRED = 3

ACTION_MATCHES_REQUIRED = 2

FACE_ACTIONS = (
    "turn_left",
    "turn_right",
    "close_left_eye",
    "close_right_eye",
    "smile",
)

_random = SystemRandom()


# ============================================================
# VERIFICATION STAGES
# ============================================================

class VerificationStage:
    ANTISPOOF = "antispoof"
    FACE_RECOGNITION = "face_recognition"
    ACTION = "action"
    DONE = "done"


# ============================================================
# VERIFICATION SESSION
# ============================================================

class VerificationSession:
    def __init__(
        self,
        reference_image: str | Path,
    ):
        self.stage = VerificationStage.ANTISPOOF

        self.reference_image = str(
            reference_image
        )

        # -----------------------------------------------
        # Anti-spoof
        # -----------------------------------------------

        self.liveness_decision: dict | None = None

        # -----------------------------------------------
        # Face recognition
        # -----------------------------------------------

        self.identity_verified = False

        self.face_match_count = 0

        # -----------------------------------------------
        # Actions
        # -----------------------------------------------

        # Random order makes replaying a predefined
        # sequence harder.
        self.actions = _random.sample(
            list(FACE_ACTIONS),
            len(FACE_ACTIONS),
        )

        self.action_index = 0

        self.action_match_count = 0

        self.completed_actions: list[str] = []

    @property
    def current_action(
        self,
    ) -> str | None:

        if self.action_index >= len(
            self.actions
        ):
            return None

        return self.actions[
            self.action_index
        ]

    def complete_current_action(
        self,
    ) -> str | None:

        action = self.current_action

        if action is None:
            return None

        self.completed_actions.append(
            action
        )

        self.action_index += 1
        self.action_match_count = 0

        return action

    @property
    def all_actions_completed(
        self,
    ) -> bool:

        return (
            self.action_index
            >= len(self.actions)
        )


# ============================================================
# LIVENESS SESSION
# ============================================================

def create_liveness_session() -> LivenessSession:
    return LivenessSession(
        window_size=settings.window_size,
        min_real_frames=(
            settings.min_real_frames
        ),
        min_avg_score=(
            settings.min_avg_score
        ),
        required_real_ratio=(
            settings.required_real_ratio
        ),
    )


def create_verification_session(
) -> VerificationSession:

    return VerificationSession(
        reference_image=REFERENCE_IMAGE,
    )


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    executor = ThreadPoolExecutor(
        max_workers=settings.max_workers,
        thread_name_prefix="verification",
    )

    inference_semaphore = asyncio.Semaphore(
        settings.max_workers
    )

    # MediaPipe FaceLandmarker instance shouldn't
    # be called concurrently from many threads.
    face_action_lock = asyncio.Lock()

    app.state.executor = executor
    app.state.inference_semaphore = (
        inference_semaphore
    )

    app.state.face_action_lock = (
        face_action_lock
    )

    print(
        "[startup]\n"
        + describe_runtime()
    )

    # --------------------------------------------------------
    # Warmup models
    # --------------------------------------------------------

    loop = asyncio.get_running_loop()

    await loop.run_in_executor(
        executor,
        warmup_antispoof,
    )

    await loop.run_in_executor(
        executor,
        warmup_deepface,
    )

    # --------------------------------------------------------
    # Face action model
    # --------------------------------------------------------

    if not FACE_LANDMARKER_MODEL.is_file():
        raise RuntimeError(
            "Face Landmarker model not found: "
            f"{FACE_LANDMARKER_MODEL}"
        )

    app.state.face_analyzer = (
        FaceActionAnalyzer(
            model_path=(
                FACE_LANDMARKER_MODEL
            ),
        )
    )

    print("[startup] warmup done")

    try:
        yield

    finally:
        print(
            "[shutdown] closing models"
        )

        try:
            app.state.face_analyzer.close()
        except Exception:
            pass

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Liveness API",
    lifespan=lifespan,
)


# ============================================================
# BLOCKING MODEL RUNNER
# ============================================================

async def run_blocking_model(
    func,
    *args,
    timeout: float,
):
    loop = asyncio.get_running_loop()

    async with app.state.inference_semaphore:

        future = loop.run_in_executor(
            app.state.executor,
            partial(
                func,
                *args,
            ),
        )

        return await asyncio.wait_for(
            future,
            timeout=timeout,
        )


# ============================================================
# FACE RECOGNITION
# ============================================================

async def process_face_recognition(
    verification_session: VerificationSession,
    frame,
    timeout: float,
) -> dict:

    verification = await run_blocking_model(
        app.state.face_analyzer.verify_face,
        verification_session.reference_image,
        frame,
        timeout=timeout,
    )

    if verification["verified"]:

        verification_session.face_match_count += 1

    else:

        # Require consecutive matches.
        verification_session.face_match_count = 0

    if (
        verification_session.face_match_count
        >= FACE_MATCHES_REQUIRED
    ):
        verification_session.identity_verified = True

        verification_session.stage = (
            VerificationStage.ACTION
        )

    return verification


# ============================================================
# FACIAL ACTION
# ============================================================

async def process_face_action(
    verification_session: VerificationSession,
    frame,
    timeout: float,
) -> dict:

    action = (
        verification_session.current_action
    )

    if action is None:
        return {
            "ok": True,
            "action": None,
            "reason": "all_actions_completed",
        }

    # Serialize calls to the MediaPipe landmarker.
    async with app.state.face_action_lock:

        result = await run_blocking_model(
            app.state.face_analyzer.check_action,
            frame,
            action,
            timeout=timeout,
        )

    if result["ok"]:

        verification_session.action_match_count += 1

    else:

        # Require consecutive detections
        verification_session.action_match_count = 0

    return result


# ============================================================
# WEBSOCKET
# ============================================================
async def send_challenge_result(
    websocket: WebSocket,
    *,
    challenge: str,
    passed: bool,
    reason: str,
    details: dict | None = None,
) -> None:
    await websocket.send_json(
        {
            "type": "challenge_result",
            "challenge": challenge,
            "passed": passed,
            "status": (
                "passed"
                if passed
                else "failed"
            ),
            "reason": reason,
            "details": details or {},
        }
    )
@app.websocket("/ws/stream")
async def ws_stream(
    websocket: WebSocket,
):
    # ========================================================
    # API KEY
    # ========================================================

    if not await check_api_key(
        websocket,
        settings.api_key,
    ):
        return

    await websocket.accept()

    # ========================================================
    # REFERENCE IMAGE
    # ========================================================

    if not REFERENCE_IMAGE.is_file():
        await websocket.send_json(
            {
                "type": "error",
                "message": (
                    "reference_image_not_found:"
                    f"{REFERENCE_IMAGE}"
                ),
            }
        )

        await websocket.close()
        return

    # ========================================================
    # SESSION STATE
    # ========================================================

    verification_session = (
        create_verification_session()
    )

    liveness_session: (
        LivenessSession | None
    ) = None

    session_id: str | None = None

    frame_counter = 0

    challenge_results: dict[
        str,
        dict,
    ] = {}

    # ========================================================
    # TIMER
    # ========================================================

    loop = asyncio.get_running_loop()

    started_at = loop.time()

    deadline = (
        started_at
        + settings.max_connection_seconds
    )

    # ========================================================
    # STORE CHALLENGE RESULT
    # ========================================================

    def save_challenge_result(
        *,
        challenge: str,
        passed: bool,
        reason: str,
        details: dict | None = None,
    ) -> None:
        challenge_results[
            challenge
        ] = {
            "challenge": challenge,
            "passed": passed,
            "status": (
                "passed"
                if passed
                else "failed"
            ),
            "reason": reason,
            "details": (
                details or {}
            ),
        }

    # ========================================================
    # CURRENT CHALLENGE
    # ========================================================

    def get_current_challenge() -> str:
        if (
            verification_session.stage
            == VerificationStage.ANTISPOOF
        ):
            return "antispoof"

        if (
            verification_session.stage
            == VerificationStage.FACE_RECOGNITION
        ):
            return "face_recognition"

        if (
            verification_session.stage
            == VerificationStage.ACTION
        ):
            return (
                verification_session.current_action
                or "action"
            )

        return "unknown"

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    async def finish_verification(
        *,
        ok: bool,
        reason: str,
    ) -> None:
        elapsed = (
            loop.time()
            - started_at
        )

        final_decision = {
            "liveness": (
                verification_session
                .liveness_decision
                is not None
            ),
            "face_verified": (
                verification_session
                .identity_verified
            ),
            "completed_actions": (
                verification_session
                .completed_actions
            ),
            "challenge_results": (
                challenge_results
            ),
        }

        await send_final(
            websocket,
            ok=ok,
            reason=reason,
            decision=final_decision,
            elapsed=elapsed,
        )

    # ========================================================
    # TIMEOUT
    # ========================================================

    async def handle_timeout() -> None:
        challenge = (
            get_current_challenge()
        )

        existing = (
            challenge_results.get(
                challenge
            )
        )

        # Do not replace an already-passed challenge.
        if not (
            existing
            and existing.get(
                "passed",
                False,
            )
        ):
            save_challenge_result(
                challenge=challenge,
                passed=False,
                reason="timeout",
                details={
                    "max_seconds": (
                        settings
                        .max_connection_seconds
                    ),
                },
            )

        await finish_verification(
            ok=False,
            reason="connection_timeout",
        )

    try:
        while True:

            # =================================================
            # CHECK TIMEOUT
            # =================================================

            remaining = (
                get_remaining_time(
                    loop,
                    deadline,
                )
            )

            if remaining <= 0:
                await handle_timeout()
                break

            # =================================================
            # RECEIVE MESSAGE
            # =================================================

            try:
                message = (
                    await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=remaining,
                    )
                )

            except asyncio.TimeoutError:
                await handle_timeout()
                break

            message_type = (
                message.get(
                    "type"
                )
            )

            # =================================================
            # CLOSE
            # =================================================

            if message_type == "close":
                break

            # =================================================
            # RESET
            # =================================================

            if message_type == "reset":
                liveness_session = None
                session_id = None
                frame_counter = 0

                challenge_results.clear()

                verification_session = (
                    create_verification_session()
                )

                await websocket.send_json(
                    {
                        "type": "reset_ok",
                        "stage": (
                            VerificationStage
                            .ANTISPOOF
                        ),
                        "remaining_seconds": round(
                            get_remaining_time(
                                loop,
                                deadline,
                            ),
                            2,
                        ),
                    }
                )

                continue

            # =================================================
            # INVALID MESSAGE
            # =================================================

            if message_type != "frame":
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "unknown_type:"
                            f"{message_type}"
                        ),
                    }
                )

                continue

            # =================================================
            # SESSION ID
            # =================================================

            incoming_session_id = str(
                message.get(
                    "session_id"
                )
                or "default"
            )

            if (
                liveness_session is None
                or incoming_session_id
                != session_id
            ):
                session_id = (
                    incoming_session_id
                )

                liveness_session = (
                    create_liveness_session()
                )

                verification_session = (
                    create_verification_session()
                )

                challenge_results.clear()

                frame_counter = 0

            # =================================================
            # DECODE BASE64
            # =================================================

            image_bytes = (
                decode_b64_image(
                    message.get(
                        "frame",
                        "",
                    )
                )
            )

            if not image_bytes:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "bad_frame_encoding"
                        ),
                    }
                )

                continue

            frame_counter += 1

            remaining = (
                get_remaining_time(
                    loop,
                    deadline,
                )
            )

            if remaining <= 0:
                await handle_timeout()
                break

            # =================================================
            # STAGE 1 — ANTI-SPOOF
            # =================================================

            if (
                verification_session.stage
                == VerificationStage.ANTISPOOF
            ):
                try:
                    result = (
                        await process_liveness_frame(
                            image_bytes=image_bytes,
                            executor=(
                                app.state.executor
                            ),
                            semaphore=(
                                app.state
                                .inference_semaphore
                            ),
                            timeout=remaining,
                        )
                    )

                except asyncio.TimeoutError:
                    await handle_timeout()
                    break

                except Exception as exc:
                    save_challenge_result(
                        challenge="antispoof",
                        passed=False,
                        reason="liveness_error",
                        details={
                            "error": str(exc),
                        },
                    )

                    await finish_verification(
                        ok=False,
                        reason="liveness_error",
                    )

                    break

                decision = (
                    liveness_session.add(
                        is_real=result[
                            "is_real"
                        ],
                        score=result[
                            "score"
                        ],
                        reason=result[
                            "reason"
                        ],
                        frame_index=(
                            frame_counter
                        ),
                    )
                )

                elapsed = (
                    loop.time()
                    - started_at
                )

                remaining = (
                    get_remaining_time(
                        loop,
                        deadline,
                    )
                )

                # ---------------------------------------------
                # PROGRESS
                # ---------------------------------------------

                await send_ack(
                    websocket,
                    frame_index=frame_counter,
                    result=result,
                    decision=decision,
                    elapsed=elapsed,
                    remaining=remaining,
                )

                # ---------------------------------------------
                # SCREEN REPLAY FAIL
                # ---------------------------------------------

                if (
                    decision.get(
                        "reason"
                    )
                    == "screen_replay_detected"
                ):
                    save_challenge_result(
                        challenge="antispoof",
                        passed=False,
                        reason=(
                            "screen_replay_detected"
                        ),
                        details={
                            "decision": (
                                decision
                            ),
                        },
                    )

                    await finish_verification(
                        ok=False,
                        reason=(
                            "screen_replay_detected"
                        ),
                    )

                    break

                # ---------------------------------------------
                # ANTI-SPOOF PASSED
                # ---------------------------------------------

                if (
                    decision.get(
                        "ready"
                    )
                    and
                    decision.get(
                        "ok"
                    )
                ):
                    verification_session.liveness_decision = (
                        decision
                    )

                    save_challenge_result(
                        challenge="antispoof",
                        passed=True,
                        reason="liveness_passed",
                        details={
                            "real_count": (
                                decision.get(
                                    "real_count"
                                )
                            ),
                            "fake_count": (
                                decision.get(
                                    "fake_count"
                                )
                            ),
                            "real_ratio": (
                                decision.get(
                                    "real_ratio"
                                )
                            ),
                            "avg_score": (
                                decision.get(
                                    "avg_score"
                                )
                            ),
                        },
                    )

                    verification_session.stage = (
                        VerificationStage
                        .FACE_RECOGNITION
                    )

                    await websocket.send_json(
                        {
                            "type": (
                                "liveness_passed"
                            ),
                            "ok": True,
                            "next_stage": (
                                "face_recognition"
                            ),
                        }
                    )

                continue

            # =================================================
            # DECODE OPENCV FRAME
            # =================================================

            frame = decode_bgr(
                image_bytes
            )

            if frame is None:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "cannot_decode_frame"
                        ),
                    }
                )

                continue

            # =================================================
            # STAGE 2 — FACE RECOGNITION
            # =================================================

            if (
                verification_session.stage
                == VerificationStage
                .FACE_RECOGNITION
            ):
                try:
                    verification = (
                        await process_face_recognition(
                            verification_session,
                            frame,
                            remaining,
                        )
                    )

                except asyncio.TimeoutError:
                    await handle_timeout()
                    break

                except ValueError:
                    verification_session.face_match_count = 0

                    await websocket.send_json(
                        {
                            "type": (
                                "face_recognition"
                            ),
                            "verified": False,
                            "identity_verified": (
                                False
                            ),
                            "reason": (
                                "face_not_detected"
                            ),
                            "match_count": 0,
                            "required_matches": (
                                FACE_MATCHES_REQUIRED
                            ),
                        }
                    )

                    continue

                except Exception as exc:
                    save_challenge_result(
                        challenge=(
                            "face_recognition"
                        ),
                        passed=False,
                        reason=(
                            "face_recognition_error"
                        ),
                        details={
                            "error": str(
                                exc
                            ),
                        },
                    )

                    await finish_verification(
                        ok=False,
                        reason=(
                            "face_recognition_error"
                        ),
                    )

                    break

                identity_verified = (
                    verification_session
                    .identity_verified
                )

                response = {
                    "type": (
                        "face_recognition"
                    ),
                    "verified": (
                        verification.get(
                            "verified",
                            False,
                        )
                    ),
                    "identity_verified": (
                        identity_verified
                    ),
                    "match_count": (
                        verification_session
                        .face_match_count
                    ),
                    "required_matches": (
                        FACE_MATCHES_REQUIRED
                    ),
                    "distance": (
                        verification.get(
                            "distance"
                        )
                    ),
                    "threshold": (
                        verification.get(
                            "threshold"
                        )
                    ),
                }

                # ---------------------------------------------
                # FACE PASSED
                # ---------------------------------------------

                if identity_verified:
                    save_challenge_result(
                        challenge=(
                            "face_recognition"
                        ),
                        passed=True,
                        reason="face_matched",
                        details={
                            "match_count": (
                                verification_session
                                .face_match_count
                            ),
                            "required_matches": (
                                FACE_MATCHES_REQUIRED
                            ),
                            "distance": (
                                verification.get(
                                    "distance"
                                )
                            ),
                            "threshold": (
                                verification.get(
                                    "threshold"
                                )
                            ),
                        },
                    )

                    response[
                        "next_stage"
                    ] = "action"

                    response[
                        "action"
                    ] = (
                        verification_session
                        .current_action
                    )

                await websocket.send_json(
                    response
                )

                continue

            # =================================================
            # STAGE 3 — ONE ACTION ONLY
            # =================================================

            if (
                verification_session.stage
                == VerificationStage.ACTION
            ):
                current_action = (
                    verification_session
                    .current_action
                )

                # ---------------------------------------------
                # NO ACTION
                # ---------------------------------------------

                if current_action is None:
                    save_challenge_result(
                        challenge="action",
                        passed=False,
                        reason=(
                            "no_action_available"
                        ),
                    )

                    await finish_verification(
                        ok=False,
                        reason=(
                            "no_action_available"
                        ),
                    )

                    break

                try:
                    action_result = (
                        await process_face_action(
                            verification_session,
                            frame,
                            remaining,
                        )
                    )

                except asyncio.TimeoutError:
                    await handle_timeout()
                    break

                except Exception as exc:
                    save_challenge_result(
                        challenge=(
                            current_action
                        ),
                        passed=False,
                        reason=(
                            "face_action_error"
                        ),
                        details={
                            "error": str(
                                exc
                            ),
                        },
                    )

                    await finish_verification(
                        ok=False,
                        reason=(
                            "face_action_error"
                        ),
                    )

                    break

                # Save count BEFORE
                # complete_current_action()
                # resets it.
                matched_count = (
                    verification_session
                    .action_match_count
                )

                # ---------------------------------------------
                # ACTION PASSED
                # ---------------------------------------------

                if (
                    action_result.get(
                        "ok",
                        False,
                    )
                    and
                    matched_count
                    >= ACTION_MATCHES_REQUIRED
                ):
                    completed_action = (
                        verification_session
                        .complete_current_action()
                    )

                    save_challenge_result(
                        challenge=(
                            completed_action
                            or current_action
                        ),
                        passed=True,
                        reason=(
                            "action_detected"
                        ),
                        details={
                            "matches": (
                                matched_count
                            ),
                            "required_matches": (
                                ACTION_MATCHES_REQUIRED
                            ),
                            "analysis": (
                                action_result.get(
                                    "analysis"
                                )
                            ),
                        },
                    )

                    # -----------------------------------------
                    # Send last action progress
                    # -----------------------------------------

                    await websocket.send_json(
                        {
                            "type": (
                                "action_result"
                            ),
                            "action": (
                                current_action
                            ),
                            "detected": True,
                            "match_count": (
                                matched_count
                            ),
                            "required_matches": (
                                ACTION_MATCHES_REQUIRED
                            ),
                            "action_completed": (
                                True
                            ),
                            "completed_action": (
                                completed_action
                                or current_action
                            ),
                            "completed_actions": (
                                verification_session
                                .completed_actions
                            ),
                            "next_action": None,
                            "analysis": (
                                action_result.get(
                                    "analysis"
                                )
                            ),
                        }
                    )

                    # -----------------------------------------
                    # ONE ACTION = DONE IMMEDIATELY
                    # -----------------------------------------

                    verification_session.stage = (
                        VerificationStage.DONE
                    )

                    await finish_verification(
                        ok=True,
                        reason=(
                            "verification_complete"
                        ),
                    )

                    break

                # ---------------------------------------------
                # ACTION NOT COMPLETE YET
                # ---------------------------------------------

                await websocket.send_json(
                    {
                        "type": (
                            "action_result"
                        ),
                        "action": (
                            current_action
                        ),
                        "detected": bool(
                            action_result.get(
                                "ok",
                                False,
                            )
                        ),
                        "match_count": (
                            matched_count
                        ),
                        "required_matches": (
                            ACTION_MATCHES_REQUIRED
                        ),
                        "action_completed": (
                            False
                        ),
                        "completed_action": None,
                        "completed_actions": (
                            verification_session
                            .completed_actions
                        ),
                        "next_action": (
                            current_action
                        ),
                        "analysis": (
                            action_result.get(
                                "analysis"
                            )
                        ),
                    }
                )

                continue

            # =================================================
            # DONE
            # =================================================

            if (
                verification_session.stage
                == VerificationStage.DONE
            ):
                break

    # ========================================================
    # CLIENT DISCONNECTED
    # ========================================================

    except WebSocketDisconnect:
        pass

    # ========================================================
    # UNEXPECTED SERVER ERROR
    # ========================================================

    except Exception as exc:
        try:
            challenge = (
                get_current_challenge()
            )

            save_challenge_result(
                challenge=challenge,
                passed=False,
                reason="server_error",
                details={
                    "error": str(
                        exc
                    ),
                },
            )

            await finish_verification(
                ok=False,
                reason="server_error",
            )

        except Exception:
            pass

    # ========================================================
    # CLOSE CONNECTION
    # ========================================================

    finally:
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
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }