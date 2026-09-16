from collections import deque
import time


class LivenessSession:
    def __init__(
        self,
        window_size: int = 20,
        min_real_frames: int = 16,
        min_avg_score: float = 0.90,
        required_real_ratio: float = 0.80,
    ):
        self.window_size = window_size
        self.min_real_frames = min_real_frames
        self.min_avg_score = min_avg_score
        self.required_real_ratio = required_real_ratio

        # Only keep latest N frames
        self.frames = deque(
            maxlen=window_size
        )

        # Number of frames received during entire session
        self.total_frames_seen = 0

        # Permanent hard-reject flag
        self.screen_replay_detected = False

        self.created_at = time.time()

    def add(
        self,
        *,
        is_real: bool,
        score: float,
        reason: str,
        frame_index: int,
    ) -> dict:

        self.total_frames_seen += 1

        frame = {
            "index": frame_index,
            "is_real": bool(is_real),
            "score": float(score),
            "reason": reason,
            "ts": time.time(),
        }

        self.frames.append(frame)

        # Screen replay is session-wide.
        # Once detected, don't allow it to disappear
        # when it leaves the sliding window.
        if reason.startswith("screen_"):
            self.screen_replay_detected = True

        return self._decision()

    def _decision(self) -> dict:
        window_frames = list(self.frames)

        n = len(window_frames)

        if n == 0:
            return {
                "ready": False,
                "ok": False,
                "reason": "no_frames",
                "total_frames": 0,
                "total_frames_seen": (
                    self.total_frames_seen
                ),
            }

        real_frames = [
            frame
            for frame in window_frames
            if frame["is_real"]
        ]

        real_count = len(real_frames)
        fake_count = n - real_count

        real_ratio = (
            real_count / n
            if n
            else 0.0
        )

        avg_score = (
            sum(
                frame["score"]
                for frame in real_frames
            )
            / real_count
            if real_count
            else 0.0
        )

        best_frame = max(
            window_frames,
            key=lambda frame: frame["score"],
        )

        result = {
            # total_frames = current sliding window
            "total_frames": n,

            # total received since connection/session started
            "total_frames_seen": (
                self.total_frames_seen
            ),

            "window_size": self.window_size,
            "real_count": real_count,
            "fake_count": fake_count,
            "real_ratio": round(
                real_ratio,
                3,
            ),
            "avg_score": round(
                avg_score,
                4,
            ),
            "best_frame_index": (
                best_frame["index"]
            ),
        }

        # -------------------------------------------------
        # HARD REJECT
        # -------------------------------------------------

        if self.screen_replay_detected:
            return {
                **result,
                "ready": True,
                "ok": False,
                "reason": (
                    "screen_replay_detected"
                ),
            }

        # -------------------------------------------------
        # COLLECT FULL WINDOW
        # -------------------------------------------------

        if n < self.window_size:
            return {
                **result,
                "ready": False,
                "ok": False,
                "reason": (
                    f"collecting:"
                    f"{n}/{self.window_size}"
                ),
            }

        # -------------------------------------------------
        # REAL FRAME COUNT
        # -------------------------------------------------

        if real_count < self.min_real_frames:
            return {
                **result,
                "ready": True,
                "ok": False,
                "reason": (
                    "not_enough_real_frames"
                ),
            }

        # -------------------------------------------------
        # REAL RATIO
        # -------------------------------------------------

        if (
            real_ratio
            < self.required_real_ratio
        ):
            return {
                **result,
                "ready": True,
                "ok": False,
                "reason": (
                    "real_ratio_too_low"
                ),
            }

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        if avg_score < self.min_avg_score:
            return {
                **result,
                "ready": True,
                "ok": False,
                "reason": (
                    "avg_score_too_low"
                ),
            }

        # -------------------------------------------------
        # PASS
        # -------------------------------------------------

        return {
            **result,
            "ready": True,
            "ok": True,
            "reason": "ok",
        }