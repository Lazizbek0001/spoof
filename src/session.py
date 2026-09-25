import asyncio
from random import SystemRandom

from fastapi import WebSocket

from app import ACTIONS, REFERENCE_IMAGE, STAGE_ACTION, STAGE_ANTISPOOF
from liveness import LivenessSession
from src.helpers import (
    antispoof,
    check_api_key,
    decode_frame,
    receive_message,
    run_in_pool,
    warmup_antispoof,
    warmup_cpu_thread,
)
from src.config import settings

class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.state = ws.app.state
        self.loop = asyncio.get_running_loop()
        self.started = self.loop.time()
        self.deadline = self.started + settings.max_connection_seconds
        self.session_id: str | None = None
        self.reset()

    def reset(self) -> None:
        self.stage = STAGE_ANTISPOOF
        self.frame_index = 0
        self.liveness = LivenessSession(
            window_size=settings.window_size,
            min_real_frames=settings.min_real_frames,
            min_avg_score=settings.min_avg_score,
            required_real_ratio=settings.required_real_ratio,
        )
        self.face_matches = 0
        self.identity_verified = False
        self.last_distance: float | None = None
        self.action = SystemRandom().choice(ACTIONS)
        self.action_matches = 0
        self.results: dict[str, dict] = {}

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - self.loop.time())

    @property
    def elapsed(self) -> float:
        return self.loop.time() - self.started

    # -- messages -------------------------------------------------------

    async def send(self, **payload) -> None:
        await self.ws.send_json(payload)

    async def final(self, ok: bool, reason: str) -> None:
        await self.send(
            type="final", ok=ok, reason=reason,
            elapsed_seconds=round(self.elapsed, 2),
            decision={
                "liveness": self.results.get("antispoof", {}).get("passed", False),
                "face_verified": self.identity_verified,
                "action": self.action,
                "challenge_results": self.results,
            },
        )

    async def fail(self, challenge: str, reason: str, **details) -> None:
        self.results[challenge] = {"passed": False, "reason": reason, "details": details}
        await self.final(False, reason)

    def passed(self, challenge: str, reason: str, **details) -> None:
        self.results[challenge] = {"passed": True, "reason": reason, "details": details}

    # -- model calls ----------------------------------------------------

    async def check_liveness(self, frame) -> dict:
        return await antispoof(frame, self.state.cpu_pool, self.state.batcher, self.remaining)

    async def check_identity(self, frame) -> dict | None:
        """None = no face detected in the frame."""
        try:
            return await run_in_pool(
                self.state.recog_pool, self.state.recognizer.verify,
                str(REFERENCE_IMAGE), frame, timeout=self.remaining,
            )
        except ValueError:
            return None

    # -- main loop ------------------------------------------------------

    async def run(self) -> None:
        try:
            while self.remaining > 0:
                message = await receive_message(self.ws, self.remaining)
                kind = message.get("type")

                if kind == "close":
                    return
                if kind == "reset":
                    self.session_id = None
                    self.reset()
                    await self.send(type="reset_ok", stage=self.stage)
                    continue
                if kind != "frame":
                    await self.send(type="error", message=f"unknown_type:{kind}")
                    continue

                # Binary frames carry no session id: keep the current one.
                sid = str(message.get("session_id") or self.session_id or "default")
                if sid != self.session_id:
                    self.session_id = sid
                    self.reset()

                frame = await run_in_pool(
                    self.state.cpu_pool, decode_frame, message.get("frame"), timeout=self.remaining
                )
                if frame is None:
                    await self.send(type="error", message="cannot_decode_frame")
                    continue

                self.frame_index += 1
                done = await (
                    self.stage_antispoof(frame) if self.stage == STAGE_ANTISPOOF
                    else self.stage_action(frame)
                )
                if done:
                    return

            await self.fail(self.stage, "timeout", max_seconds=settings.max_connection_seconds)
        except asyncio.TimeoutError:
            await self.fail(self.stage, "timeout", max_seconds=settings.max_connection_seconds)

    # -- stage 1: anti-spoof + identity on the same frames ---------------

    async def stage_antispoof(self, frame) -> bool:
        need_identity = not self.identity_verified
        try:
            live, identity = await asyncio.gather(
                self.check_liveness(frame),
                self.check_identity(frame) if need_identity else asyncio.sleep(0),
            )
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            await self.fail("antispoof", "liveness_error", error=str(exc))
            return True

        decision = self.liveness.add(
            is_real=live["is_real"], score=live["score"],
            reason=live["reason"], frame_index=self.frame_index,
        )

        # Identity counts only on frames the anti-spoof model calls real,
        # so a photo of the victim cannot supply the matches.
        if need_identity:
            if live["is_real"] and identity and identity["verified"]:
                self.face_matches += 1
                self.last_distance = identity["distance"]
            else:
                self.face_matches = 0
            if self.face_matches >= settings.face_matches_required:
                self.identity_verified = True
                self.passed("face_recognition", "face_matched",
                            matches=self.face_matches, distance=self.last_distance)

        await self.send(
            type="ack", frame_index=self.frame_index,
            is_real=live["is_real"], score=live["score"], reason=live["reason"],
            face_matches=self.face_matches, face_verified=self.identity_verified,
            decision=decision, remaining_seconds=round(self.remaining, 2),
        )

        if decision.get("reason") == "screen_replay_detected":
            await self.fail("antispoof", "screen_replay_detected", decision=decision)
            return True

        if decision.get("ready") and decision.get("ok"):
            self.passed("antispoof", "liveness_passed",
                        **{k: decision.get(k) for k in ("real_count", "fake_count", "real_ratio", "avg_score")})
            if not self.identity_verified:
                await self.fail("face_recognition", "face_not_matched", matches=self.face_matches)
                return True
            self.stage = STAGE_ACTION
            await self.send(type="liveness_passed", ok=True, next_stage=STAGE_ACTION, action=self.action)

        return False

    # -- stage 2: one challenge ------------------------------------------

    async def stage_action(self, frame) -> bool:
        try:
            result = await run_in_pool(
                self.state.cpu_pool, self.state.face_analyzer.check_action,
                frame, self.action, timeout=self.remaining,
            )
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            await self.fail(self.action, "face_action_error", error=str(exc))
            return True

        self.action_matches = self.action_matches + 1 if result["ok"] else 0
        completed = self.action_matches >= settings.action_matches_required
        reason = result.get("reason")

        # The person doing the action must still be the recognised one.
        if completed:
            identity = await self.check_identity(frame)
            if not (identity and identity["verified"]):
                completed, reason, self.action_matches = False, "identity_mismatch", 0

        await self.send(
            type="action_result", action=self.action, detected=bool(result["ok"]),
            reason=reason, match_count=self.action_matches,
            required_matches=settings.action_matches_required,
            action_completed=completed, analysis=result.get("analysis"),
        )
        if not completed:
            return False

        self.passed(self.action, "action_detected", matches=self.action_matches)
        await self.final(True, "verification_complete")
        return True


