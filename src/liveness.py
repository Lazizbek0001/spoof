from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class FrameResult:
    index: int
    is_real: bool
    score: float
    reason: str
    timestamp: float


class LivenessSession:
    def __init__(
        self,
        *,
        min_frames: int = 12,
        min_real_frames: int = 8,
        min_avg_score: float = 0.90,
        required_real_ratio: float = 0.80,
    ) -> None:
        self.min_frames = min_frames
        self.min_real_frames = min_real_frames
        self.min_avg_score = min_avg_score
        self.required_real_ratio = required_real_ratio

        self.frames: list[FrameResult] = []

    def add(
        self,
        *,
        is_real: bool,
        score: float,
        reason: str,
        frame_index: int,
    ) -> dict:
        self.frames.append(
            FrameResult(
                index=frame_index,
                is_real=is_real,
                score=float(score),
                reason=reason,
                timestamp=time.monotonic(),
            )
        )

        return self._decision()

    def _result(
        self,
        *,
        ready: bool,
        ok: bool,
        reason: str,
        real_count: int,
        fake_count: int,
        real_ratio: float,
        avg_score: float,
        best_frame_index: int,
    ) -> dict:
        return {
            "ready": ready,
            "ok": ok,
            "reason": reason,
            "total_frames": len(self.frames),
            "real_count": real_count,
            "fake_count": fake_count,
            "real_ratio": round(real_ratio, 3),
            "avg_score": round(avg_score, 4),
            "best_frame_index": best_frame_index,
        }

    def _decision(self) -> dict:
        total = len(self.frames)

        if total == 0:
            return {
                "ready": False,
                "ok": False,
                "reason": "no_frames",
                "total_frames": 0,
            }

        real_frames = [
            frame
            for frame in self.frames
            if frame.is_real
        ]

        real_count = len(real_frames)
        fake_count = total - real_count

        real_ratio = real_count / total

        avg_score = (
            sum(frame.score for frame in real_frames)
            / real_count
            if real_count
            else 0.0
        )

        best_frame = max(
            self.frames,
            key=lambda frame: frame.score,
        )

        # Hard replay/screen rejection
        if any(
            frame.reason.startswith("screen_")
            for frame in self.frames
        ):
            return self._result(
                ready=True,
                ok=False,
                reason="screen_replay_detected",
                real_count=real_count,
                fake_count=fake_count,
                real_ratio=real_ratio,
                avg_score=avg_score,
                best_frame_index=best_frame.index,
            )

        # Still collecting
        if total < self.min_frames:
            return self._result(
                ready=False,
                ok=False,
                reason=f"collecting:{total}/{self.min_frames}",
                real_count=real_count,
                fake_count=fake_count,
                real_ratio=real_ratio,
                avg_score=avg_score,
                best_frame_index=best_frame.index,
            )

        if real_count < self.min_real_frames:
            return self._result(
                ready=True,
                ok=False,
                reason="not_enough_real_frames",
                real_count=real_count,
                fake_count=fake_count,
                real_ratio=real_ratio,
                avg_score=avg_score,
                best_frame_index=best_frame.index,
            )

        if real_ratio < self.required_real_ratio:
            return self._result(
                ready=True,
                ok=False,
                reason="real_ratio_too_low",
                real_count=real_count,
                fake_count=fake_count,
                real_ratio=real_ratio,
                avg_score=avg_score,
                best_frame_index=best_frame.index,
            )

        if avg_score < self.min_avg_score:
            return self._result(
                ready=True,
                ok=False,
                reason="avg_score_too_low",
                real_count=real_count,
                fake_count=fake_count,
                real_ratio=real_ratio,
                avg_score=avg_score,
                best_frame_index=best_frame.index,
            )

        return self._result(
            ready=True,
            ok=True,
            reason="ok",
            real_count=real_count,
            fake_count=fake_count,
            real_ratio=real_ratio,
            avg_score=avg_score,
            best_frame_index=best_frame.index,
        )