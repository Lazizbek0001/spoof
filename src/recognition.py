from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

# GPU/TF env configuration must be applied before DeepFace/TensorFlow load.
from face_util import gpu_utils as _gpu_utils  # noqa: F401

from deepface import DeepFace


# DeepFace's own cosine thresholds, used only if find_threshold is unavailable.
_FALLBACK_THRESHOLDS = {
    "Facenet512": 0.30,
    "Facenet": 0.40,
    "ArcFace": 0.68,
    "VGG-Face": 0.68,
}


class FaceRecognizer:
    """
    1:1 face verification on top of DeepFace.

    The old code called DeepFace.verify(reference_path, frame) per frame, which
    re-read the reference from disk and ran detection + embedding on it every
    time. Here the reference embedding is computed once and cached, so each
    frame costs one detection + one embedding, then a dot product.

    Same model / detector / metric / threshold as before, so decisions and
    stored Facenet512 embeddings stay compatible.
    """

    def __init__(
        self,
        *,
        model_name: str = "Facenet512",
        detector_backend: str = "retinaface",
        threshold: float | None = None,
        cache_size: int = 512,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = "cosine"
        self.threshold = (
            float(threshold) if threshold is not None else self._default_threshold()
        )

        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()

    def _default_threshold(self) -> float:
        try:
            from deepface.modules import verification

            return float(
                verification.find_threshold(self.model_name, self.distance_metric)
            )
        except Exception:
            return _FALLBACK_THRESHOLDS.get(self.model_name, 0.30)

    # ------------------------------------------------------------------

    def warmup(self) -> None:
        try:
            DeepFace.represent(
                img_path=np.zeros((160, 160, 3), dtype=np.uint8),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception as exc:
            print(f"[warmup] DeepFace failed: {exc}")

    def embed(self, image: np.ndarray | str) -> np.ndarray:
        """
        L2-normalised embedding of the largest face.
        Raises ValueError when no face is found (same as DeepFace.verify).
        """
        faces = DeepFace.represent(
            img_path=image,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=True,
            align=True,
        )

        if not faces:
            raise ValueError("face_not_detected")

        def area(face: dict) -> int:
            box = face.get("facial_area") or {}
            return int(box.get("w", 0)) * int(box.get("h", 0))

        vector = np.asarray(max(faces, key=area)["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(vector))

        if norm <= 0.0:
            raise ValueError("face_not_detected")

        return vector / norm

    def reference_embedding(self, reference: str | Path) -> np.ndarray:
        """Cached by path + mtime + size, so a replaced photo is picked up."""
        path = str(reference)
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)

        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        vector = self.embed(path)

        with self._cache_lock:
            self._cache[key] = vector
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return vector

    def verify(
        self,
        reference: str | Path | np.ndarray,
        frame_bgr: np.ndarray,
    ) -> dict[str, Any]:
        """
        `reference` is an image path or an already computed embedding.
        Raises ValueError when the frame has no detectable face.
        """
        if isinstance(reference, np.ndarray) and reference.ndim == 1:
            ref = reference.astype(np.float32)
            ref = ref / max(float(np.linalg.norm(ref)), 1e-12)
        else:
            ref = self.reference_embedding(reference)

        distance = 1.0 - float(np.dot(ref, self.embed(frame_bgr)))

        return {
            "verified": distance <= self.threshold,
            "distance": distance,
            "threshold": self.threshold,
            "model": self.model_name,
            "detector": self.detector_backend,
            "metric": self.distance_metric,
        }