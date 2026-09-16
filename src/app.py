from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)

from src.config import settings
from src.helpers import (
    check_api_key,
    decode_b64_image,
    get_remaining_time,
    process_liveness_frame,
    send_ack,
    send_final,
    send_timeout,
    warmup_antispoof,
    warmup_deepface,
)
from src.liveness import LivenessSession

from face_util.gpu_utils import (
    describe_runtime,
    get_runtime_info,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    executor = ThreadPoolExecutor(
        max_workers=settings.max_workers,
        thread_name_prefix="liveness",
    )

    inference_semaphore = asyncio.Semaphore(
        settings.max_workers
    )

    app.state.executor = executor
    app.state.inference_semaphore = (
        inference_semaphore
    )

    print(
        "[startup]\n"
        + describe_runtime()
    )

    loop = asyncio.get_running_loop()

    # Sequential warmup is more predictable when
    # PyTorch and TensorFlow share the same machine/GPU.
    await loop.run_in_executor(
        executor,
        warmup_antispoof,
    )

    await loop.run_in_executor(
        executor,
        warmup_deepface,
    )

    print("[startup] warmup done")

    try:
        yield

    finally:
        print(
            "[shutdown] closing executor"
        )

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


app = FastAPI(
    title="Liveness API",
    lifespan=lifespan,
)


def create_session() -> LivenessSession:
    return LivenessSession(
        min_frames=settings.min_frames,
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


@app.websocket("/ws/stream")
async def ws_stream(
    websocket: WebSocket,
):
    # ---------------------------------------------------------
    # API KEY
    # ---------------------------------------------------------

    if not await check_api_key(
        websocket,
        settings.api_key,
    ):
        return

    await websocket.accept()

    loop = asyncio.get_running_loop()

    started_at = loop.time()

    deadline = (
        started_at
        + settings.max_connection_seconds
    )

    session: LivenessSession | None = None
    session_id: str | None = None

    frame_counter = 0

    try:
        while True:
            # -------------------------------------------------
            # Connection timeout
            # -------------------------------------------------

            remaining = get_remaining_time(
                loop,
                deadline,
            )

            if remaining <= 0:
                await send_timeout(
                    websocket,
                    max_seconds=(
                        settings
                        .max_connection_seconds
                    ),
                )
                break

            # -------------------------------------------------
            # Receive message
            # -------------------------------------------------

            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=remaining,
                )

            except asyncio.TimeoutError:
                await send_timeout(
                    websocket,
                    max_seconds=(
                        settings
                        .max_connection_seconds
                    ),
                )
                break

            message_type = message.get("type")

            # -------------------------------------------------
            # Client requested close
            # -------------------------------------------------

            if message_type == "close":
                break

            if message_type == "reset":
                session = None
                session_id = None
                frame_counter = 0

                await websocket.send_json(
                    {
                        "type": "reset_ok",
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

            # -------------------------------------------------
            # Validate message
            # -------------------------------------------------

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

            incoming_session_id = str(
                message.get("session_id")
                or "default"
            )

            # New ID = new liveness attempt
            if (
                session is None
                or incoming_session_id
                != session_id
            ):
                session_id = (
                    incoming_session_id
                )

                session = create_session()

                frame_counter = 0

            # -------------------------------------------------
            # Decode frame
            # -------------------------------------------------

            image_bytes = decode_b64_image(
                message.get("frame", "")
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

            # -------------------------------------------------
            # Check remaining time again
            # -------------------------------------------------

            remaining = get_remaining_time(
                loop,
                deadline,
            )

            if remaining <= 0:
                await send_timeout(
                    websocket,
                    max_seconds=(
                        settings
                        .max_connection_seconds
                    ),
                )
                break

            # -------------------------------------------------
            # Inference
            # -------------------------------------------------

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
                await send_timeout(
                    websocket,
                    max_seconds=(
                        settings
                        .max_connection_seconds
                    ),
                )
                break

            except Exception as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "liveness_error:"
                            f"{exc}"
                        ),
                    }
                )
                continue

            # -------------------------------------------------
            # Liveness decision
            # -------------------------------------------------

            decision = session.add(
                is_real=result["is_real"],
                score=result["score"],
                reason=result["reason"],
                frame_index=frame_counter,
            )

            elapsed = (
                loop.time()
                - started_at
            )

            remaining = get_remaining_time(
                loop,
                deadline,
            )

            # -------------------------------------------------
            # Frame result
            # -------------------------------------------------

            await send_ack(
                websocket,
                frame_index=frame_counter,
                result=result,
                decision=decision,
                elapsed=elapsed,
                remaining=remaining,
            )

            # -------------------------------------------------
            # Hard screen/replay rejection
            # -------------------------------------------------

            if (
                decision.get("reason")
                == "screen_replay_detected"
            ):
                await send_final(
                    websocket,
                    ok=False,
                    reason=(
                        "screen_replay_detected"
                    ),
                    decision=decision,
                    elapsed=elapsed,
                )
                break

            # -------------------------------------------------
            # Successful liveness
            # -------------------------------------------------

            if (
                decision.get("ready")
                and decision.get("ok")
            ):
                await send_final(
                    websocket,
                    ok=True,
                    reason="liveness_passed",
                    decision=decision,
                    elapsed=elapsed,
                )
                break

    except WebSocketDisconnect:
        pass

    except Exception as exc:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "server_error:"
                        f"{exc}"
                    ),
                }
            )
        except Exception:
            pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health():
    return {
        "ok": True,
        "runtime": get_runtime_info(),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }