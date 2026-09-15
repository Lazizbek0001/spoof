from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

# Silence DeepFace/TF noise before importing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from deepface import DeepFace

from .gpu_utils import get_runtime_info


FACE_DIM = 512  # ArcFace embedding size
FACE_ONLY_ERROR_MSG = "Yuz mos kelmadi."

# Cosine-distance threshold for ArcFace.
# DeepFace's own verification threshold for ArcFace is ~0.68 (cosine).
# We use a slightly stricter one for auth-style use.
ARCFACE_COSINE_THRESHOLD = 0.55

# Detector backend: "retinaface" (accurate) or "opencv" (fast).
# Change to "opencv" if you need lower latency on CPU.
DETECTOR_BACKEND = "retinaface"

# Recognizer model. Options: "ArcFace", "Facenet512", "SFace", "VGG-Face".
RECOGNIZER_MODEL = "Facenet512"  # ArcFace is default in DeepFace, but Facenet512 is faster and smaller


def _decode_image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot_decode_input")
    rgb = bgr[:, :, ::-1]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _represent(image_rgb: np.ndarray) -> np.ndarray | None:
    """
    Returns a normalized 512-d embedding, or None if no face found.
    """
    try:
        reps = DeepFace.represent(
            img_path=image_rgb,
            model_name=RECOGNIZER_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
            normalization="base",  # let DeepFace handle ArcFace normalization
        )
    except Exception:
        return None

    if not reps:
        return None

    # DeepFace returns a list of dicts; take the largest / first face
    # (represent already picks the best by default when multiple are present
    # only if you pass detector; otherwise it returns all).
    # We just take the first.
    emb = reps[0].get("embedding")
    if emb is None:
        return None

    vec = np.asarray(emb, dtype=np.float32)
    if vec.shape != (FACE_DIM,):
        # If the model returns a different dim (e.g. Facenet512 = 512, SFace = 128)
        # we still accept it as long as it's 1-D.
        if vec.ndim != 1 or vec.size == 0:
            return None

    # L2-normalize for cosine distance
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def _profile_cached_encoding(user) -> np.ndarray | None:
    profile = getattr(user, "profile", None)
    if not profile:
        return None

    raw = getattr(profile, "face_encoding", None)
    if not raw:
        return None

    try:
        vec = np.frombuffer(raw, dtype=np.float32).copy()
        if vec.ndim != 1 or vec.size == 0:
            return None
        # Re-normalize in case it was stored unnormalized
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec
    except Exception:
        return None


def _profile_photo_path(user) -> str | None:
    profile = getattr(user, "profile", None)
    if not profile:
        return None

    photo = getattr(profile, "photo", None)
    if not photo:
        return None

    try:
        p = photo.path
        if p and os.path.exists(p):
            return p
    except Exception:
        return None

    return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Both are already L2-normalized, so cosine distance = 1 - dot(a, b)
    return float(1.0 - np.dot(a, b))


def verify_face_only(user, image_bytes: bytes, debug: bool = False) -> Tuple[bool, str, str]:
    """
    Returns (ok, user_message, debug_reason).
    Uses DeepFace (ArcFace) with cosine distance.
    """
    try:
        runtime = get_runtime_info()

        unknown_rgb = _decode_image_bytes_to_rgb(image_bytes)
        unknown_enc = _represent(unknown_rgb)
        if unknown_enc is None:
            return False, "Yuz aniqlanmadi. Kameraga yaqinroq keling.", "no_face_in_input"

        ref_enc = _profile_cached_encoding(user)

        if ref_enc is None:
            ref_path = _profile_photo_path(user)
            if not ref_path:
                return False, "Profil rasm topilmadi. Admin profilga rasm yuklasin.", "no_profile_photo"

            ref_bgr = cv2.imread(ref_path)
            if ref_bgr is None:
                return False, "Profil rasm topilmadi. Admin profilga rasm yuklasin.", "no_profile_photo"

            ref_rgb = np.ascontiguousarray(ref_bgr[:, :, ::-1], dtype=np.uint8)
            ref_enc = _represent(ref_rgb)
            if ref_enc is None:
                return False, "Profil rasmda yuz topilmadi. Profil rasmini yangilang.", "no_face_in_profile_photo"

        # Dimension mismatch check (e.g. old 128-d cached encodings vs new 512-d)
        if ref_enc.shape != unknown_enc.shape:
            return (
                False,
                "Profil rasmini yangilash kerak. Eski formatdagi ma'lumot.",
                f"dim_mismatch:ref={ref_enc.shape},unk={unknown_enc.shape}",
            )

        dist = _cosine_distance(ref_enc, unknown_enc)

        if dist > ARCFACE_COSINE_THRESHOLD:
            return (
                False,
                FACE_ONLY_ERROR_MSG,
                f"face_mismatch:cos={dist:.4f},device={runtime['device']}"
                if debug
                else "face_mismatch",
            )

        return (
            True,
            "ok",
            f"face_ok:cos={dist:.4f},device={runtime['device']}"
            if debug
            else "face_ok",
        )

    except Exception as e:
        return False, "Boshqattan harakat qilib ko‘ring!", f"face_verify_exception:{e}"


def compute_and_serialize_encoding(user, image_bytes: bytes) -> bytes | None:
    """
    Helper for enrollment: given a user and a raw image, compute the ArcFace
    embedding and return it as raw float32 bytes ready to store in
    user.profile.face_encoding.

    Returns None if no face was detected.
    """
    try:
        rgb = _decode_image_bytes_to_rgb(image_bytes)
        vec = _represent(rgb)
        if vec is None:
            return None
        return vec.astype(np.float32).tobytes()
    except Exception:
        return None