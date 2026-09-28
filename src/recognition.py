from __future__ import annotations

from typing import Any

import numpy as np

from face_util import gpu_utils as _gpu_utils  # noqa: F401  (must load before TF)
from deepface import DeepFace


class FaceRecognizer:
    """
    Facenet512 embeddings via DeepFace. The reference is a vector supplied by
    the client (produced by POST /embed or by the same model elsewhere), so the
    server never stores photos.
    """

    MODEL = "Facenet512"
    DETECTOR = "retinaface"
    DIM = 512

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = float(threshold) if threshold is not None else self._default_threshold()

    def _default_threshold(self) -> float:
        try:
            from deepface.modules import verification
            return float(verification.find_threshold(self.MODEL, "cosine"))
        except Exception:
            return 0.30  # DeepFace's cosine threshold for Facenet512

    def warmup(self) -> None:
        try:
            DeepFace.represent(
                img_path=np.zeros((160, 160, 3), np.uint8),
                model_name=self.MODEL, detector_backend=self.DETECTOR, enforce_detection=False,
            )
        except Exception as exc:
            print(f"[warmup] DeepFace failed: {exc}")

    @staticmethod
    def normalize(vector) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("invalid_embedding")
        return v / norm

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """L2-normalised embedding of the largest face. ValueError if no face."""
        faces = DeepFace.represent(
            img_path=image_bgr, model_name=self.MODEL, detector_backend=self.DETECTOR,
            enforce_detection=True, align=True,
        )
        if not faces:
            raise ValueError("face_not_detected")

        def area(face: dict) -> int:
            box = face.get("facial_area") or {}
            return int(box.get("w", 0)) * int(box.get("h", 0))

        return self.normalize(max(faces, key=area)["embedding"])

    def verify(self, references: np.ndarray, frame_bgr: np.ndarray) -> dict[str, Any]:
        """
        references: (N, DIM) normalised vectors; the closest one decides.
        ValueError if the frame has no detectable face.
        """
        probe = self.embed(frame_bgr)
        distance = float(np.min(1.0 - references @ probe))
        return {"verified": distance <= self.threshold, "distance": distance, "threshold": self.threshold}