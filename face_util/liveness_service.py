from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .anti_spoof_infer import predict_antispoof_score


@dataclass
class FrameLivenessResult:
    index: int
    is_real: bool
    score: float
    reason: str


@dataclass
class MultiFrameLivenessResult:
    ok: bool
    avg_score: float
    real_count: int
    fake_count: int
    best_frame_index: int
    details: List[FrameLivenessResult]
    reason: str


def decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def check_liveness_from_frames(
    frames_bytes: List[bytes],
    min_real_frames: int = 2,
    min_avg_score: float = 0.60,
) -> MultiFrameLivenessResult:
    details: List[FrameLivenessResult] = []
    screen_hits = 0

    real_scores: List[float] = []
    best_frame_index = -1
    best_score = -1.0

    # --- optional: reject very static frame sequence ---
    motion_scores: List[float] = []

    def _frame_diff_score(a: bytes, b: bytes) -> float:
        img1 = decode_image_bytes(a)
        img2 = decode_image_bytes(b)

        if img1 is None or img2 is None:
            return 0.0

        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        if h < 8 or w < 8:
            return 0.0

        g1 = cv2.resize(g1, (w, h))
        g2 = cv2.resize(g2, (w, h))

        diff = cv2.absdiff(g1, g2)
        return float(diff.mean())

    for idx, raw in enumerate(frames_bytes):
        img = decode_image_bytes(raw)
        if img is None:
            details.append(
                FrameLivenessResult(
                    index=idx,
                    is_real=False,
                    score=0.0,
                    reason="decode_failed",
                )
            )
            continue

        is_real, score, reason = predict_antispoof_score(img)

        if reason.startswith("screen_"):
            screen_hits += 1

        details.append(
            FrameLivenessResult(
                index=idx,
                is_real=is_real,
                score=score,
                reason=reason,
            )
        )

        if is_real:
            real_scores.append(score)
            if score > best_score:
                best_score = score
                best_frame_index = idx

    # --- motion check between neighboring frames ---
    if len(frames_bytes) >= 2:
        for i in range(len(frames_bytes) - 1):
            try:
                motion_scores.append(_frame_diff_score(frames_bytes[i], frames_bytes[i + 1]))
            except Exception:
                motion_scores.append(0.0)

    real_count = sum(1 for x in details if x.is_real)
    fake_count = len(details) - real_count
    avg_score = float(sum(real_scores) / len(real_scores)) if real_scores else 0.0

    if not details:
        return MultiFrameLivenessResult(
            ok=False,
            avg_score=0.0,
            real_count=0,
            fake_count=0,
            best_frame_index=-1,
            details=[],
            reason="no_frames",
        )

    # hard reject if screen replay heuristic fired on any frame
    if screen_hits >= 1:
        return MultiFrameLivenessResult(
            ok=False,
            avg_score=avg_score,
            real_count=real_count,
            fake_count=fake_count,
            best_frame_index=best_frame_index,
            details=details,
            reason="screen_replay_detected",
        )

    # reject if frames are too static
    if motion_scores and all(m < 1.5 for m in motion_scores):
        return MultiFrameLivenessResult(
            ok=False,
            avg_score=avg_score,
            real_count=real_count,
            fake_count=fake_count,
            best_frame_index=best_frame_index,
            details=details,
            reason="too_static_possible_replay",
        )

    if real_count < min_real_frames:
        return MultiFrameLivenessResult(
            ok=False,
            avg_score=avg_score,
            real_count=real_count,
            fake_count=fake_count,
            best_frame_index=best_frame_index,
            details=details,
            reason="not_enough_real_frames",
        )

    if avg_score < min_avg_score:
        return MultiFrameLivenessResult(
            ok=False,
            avg_score=avg_score,
            real_count=real_count,
            fake_count=fake_count,
            best_frame_index=best_frame_index,
            details=details,
            reason="avg_score_too_low",
        )

    return MultiFrameLivenessResult(
        ok=True,
        avg_score=avg_score,
        real_count=real_count,
        fake_count=fake_count,
        best_frame_index=best_frame_index,
        details=details,
        reason="ok",
    )

def _decode_gray(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def _crop_safe(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h < 8 or w < 8:
        return 0.0
    a = cv2.resize(a, (w, h))
    b = cv2.resize(b, (w, h))
    return float(cv2.absdiff(a, b).mean())

def detect_rigid_screen_motion(frames_bytes: list[bytes]) -> tuple[bool, str]:
    if len(frames_bytes) < 2:
        return False, "not_enough_frames_for_rigid_check"

    imgs = []
    for raw in frames_bytes[:3]:
        img = _decode_gray(raw)
        if img is None:
            return False, "decode_failed"
        imgs.append(img)

    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    pair_scores = []

    for i in range(len(imgs) - 1):
        a = imgs[i]
        b = imgs[i + 1]

        faces = face.detectMultiScale(a, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

        face_a = _crop_safe(a, x, y, x + w, y + h)
        face_b = _crop_safe(b, x, y, x + w, y + h)

        bg_mask_a = a.copy()
        bg_mask_b = b.copy()

        bg_mask_a[y:y + h, x:x + w] = 0
        bg_mask_b[y:y + h, x:x + w] = 0

        face_diff = _mean_abs_diff(face_a, face_b)
        bg_diff = _mean_abs_diff(bg_mask_a, bg_mask_b)

        if face_diff <= 0.01:
            continue

        ratio = bg_diff / max(face_diff, 1e-6)
        pair_scores.append((face_diff, bg_diff, ratio))

    if not pair_scores:
        return False, "rigid_check_insufficient_data"

    suspicious_pairs = 0
    strong_pairs = 0

    for face_diff, bg_diff, ratio in pair_scores:
        if ratio > 0.75 and bg_diff > 2.5:
            suspicious_pairs += 1
        if ratio > 0.90 and bg_diff > 4.0:
            strong_pairs += 1

    # hard reject only if evidence is really strong
    if strong_pairs >= 1 or suspicious_pairs >= 2:
        return True, f"rigid_screen_motion:{pair_scores}"

    return False, f"motion_ok:{pair_scores}"