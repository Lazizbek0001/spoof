from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .gpu_utils import get_antispoof_runtime_info
from .src.anti_spoof_predict import AntiSpoofPredict
from .src.generate_patches import CropImage


MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "resources",
    "anti_spoof_models",
)

# Original MiniVision pretrained weights: class 1 = REAL / LIVE.
REAL_LABEL = 1

# Set ANTISPOOF_FP16=0 to disable half precision on the GPU.
USE_FP16 = os.getenv("ANTISPOOF_FP16", "1").strip().lower() not in ("0", "false", "no")

Result = Tuple[bool, float, str]


# ---------------------------------------------------------------------------
# Cached model objects
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_antispoof_model() -> AntiSpoofPredict:
    """
    Build the predictor once: GPU when CUDA is available, CPU otherwise.
    All .pth models are loaded here, not per frame.
    """
    runtime = get_antispoof_runtime_info()
    on_gpu = runtime["device"] == "GPU"

    predictor = AntiSpoofPredict(
        device_id=runtime["device_id"] or 0,
        force_cpu=not on_gpu,
        model_dir=MODEL_DIR,
        use_fp16=USE_FP16,
    )

    print(
        f"[Anti-Spoof] device={predictor.device} "
        f"fp16={predictor.use_fp16} "
        f"gpu={runtime.get('gpu_name')} "
        f"models={[m.name for m in predictor.models]}"
    )

    if runtime.get("error"):
        print(f"[Anti-Spoof] runtime error: {runtime['error']}")

    return predictor


@lru_cache(maxsize=1)
def get_cropper() -> CropImage:
    return CropImage()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _safe_resize_if_needed(
    image_bgr: np.ndarray,
    max_side: int = 1280,
) -> np.ndarray:
    """
    Reduce very large images before running face detection / anti-spoof.

    Keeps the aspect ratio unchanged.
    """
    h, w = image_bgr.shape[:2]

    if max(h, w) <= max_side:
        return image_bgr

    scale = max_side / float(max(h, w))

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(
        image_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


# ---------------------------------------------------------------------------
# Screen / replay heuristic
# ---------------------------------------------------------------------------

def _looks_like_screen_replay(
    img_bgr: np.ndarray,
) -> tuple[bool, str]:
    """
    Lightweight screen/replay heuristic.

    Looks for:
        - repetitive banding
        - moire-like high frequency information
        - unusually strong edge / reflection patterns

    Returns:
        (is_screen_suspected, reason)
    """

    try:
        if img_bgr is None or img_bgr.size == 0:
            return False, "empty"

        h, w = img_bgr.shape[:2]

        # Downscale because this heuristic does not need
        # full-resolution images.
        scale = 320.0 / max(h, w)

        if scale < 1.0:
            img = cv2.resize(
                img_bgr,
                (
                    max(1, int(w * scale)),
                    max(1, int(h * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img = img_bgr.copy()

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY,
        )

        # ---------------------------------------------------------------
        # Laplacian energy
        # ---------------------------------------------------------------

        lap = cv2.Laplacian(
            gray,
            cv2.CV_32F,
        )

        lap_abs = np.abs(lap)

        col_energy = lap_abs.mean(axis=0)
        row_energy = lap_abs.mean(axis=1)

        col_std = float(np.std(col_energy))
        row_std = float(np.std(row_energy))

        # ---------------------------------------------------------------
        # Frequency-domain score
        # ---------------------------------------------------------------

        fft = np.fft.fft2(
            gray.astype(np.float32)
        )

        fft_shift = np.fft.fftshift(fft)

        magnitude = np.log1p(
            np.abs(fft_shift)
        )

        mh, mw = magnitude.shape

        cy = mh // 2
        cx = mw // 2

        mask = np.ones_like(
            magnitude,
            dtype=np.float32,
        )

        radius = max(
            8,
            min(mh, mw) // 12,
        )

        y1 = max(0, cy - radius)
        y2 = min(mh, cy + radius)

        x1 = max(0, cx - radius)
        x2 = min(mw, cx + radius)

        mask[y1:y2, x1:x2] = 0.0

        hf_score = float(
            (magnitude * mask).mean()
        )

        # ---------------------------------------------------------------
        # Edge density
        # ---------------------------------------------------------------

        edges = cv2.Canny(
            gray,
            80,
            180,
        )

        line_ratio = float(
            edges.mean() / 255.0
        )

        # ---------------------------------------------------------------
        # Screen suspicion rules
        # ---------------------------------------------------------------

        if (
            col_std > 14.0
            and hf_score > 2.2
        ) or (
            row_std > 14.0
            and hf_score > 2.2
        ):
            return (
                True,
                (
                    "screen_banding:"
                    f"col_std={col_std:.2f},"
                    f"row_std={row_std:.2f},"
                    f"hf={hf_score:.2f}"
                ),
            )

        if (
            line_ratio > 0.11
            and hf_score > 2.15
        ):
            return (
                True,
                (
                    "screen_reflection:"
                    f"line_ratio={line_ratio:.3f},"
                    f"hf={hf_score:.2f}"
                ),
            )

        return (
            False,
            (
                "ok_screen_check:"
                f"col_std={col_std:.2f},"
                f"row_std={row_std:.2f},"
                f"hf={hf_score:.2f},"
                f"line={line_ratio:.3f}"
            ),
        )

    except Exception as exc:
        return (
            False,
            f"screen_check_error:{exc}",
        )


# ---------------------------------------------------------------------------
# Anti-spoof pipeline, split so the GPU part can be batched across sessions
#
#   prepare_antispoof()      CPU, per frame, any worker thread
#   infer_antispoof_batch()  GPU, one call for many frames, single GPU thread
#   finalize_antispoof()     trivial, per frame
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AntiSpoofInput:
    # One crop per loaded model, each (H, W, 3) uint8 BGR.
    crops: List[np.ndarray]
    is_screen: bool
    screen_reason: str


def prepare_antispoof(
    image_bgr: np.ndarray,
) -> Tuple[Optional[AntiSpoofInput], Optional[Result]]:
    """
    CPU work: resize, face bbox, screen heuristic, per-model crops.

    Returns (input, None) when the frame should go to the model,
    or (None, result) when the answer is already known (no face, ...).
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, (False, 0.0, "empty_image")

    image_bgr = _safe_resize_if_needed(image_bgr)

    predictor = get_antispoof_model()

    try:
        bbox = predictor.get_bbox(image_bgr)
    except Exception as exc:
        return None, (False, 0.0, f"bbox_error:{exc}")

    if not bbox:
        return None, (False, 0.0, "no_face_bbox")

    is_screen, screen_reason = _looks_like_screen_replay(image_bgr)

    cropper = get_cropper()
    crops: List[np.ndarray] = []

    for loaded in predictor.models:
        crops.append(
            cropper.crop(
                org_img=image_bgr,
                bbox=bbox,
                scale=1.0 if loaded.scale is None else loaded.scale,
                out_w=loaded.w_input,
                out_h=loaded.h_input,
                crop=loaded.scale is not None,
            )
        )

    return AntiSpoofInput(crops, is_screen, screen_reason), None


def infer_antispoof_batch(
    inputs: Sequence[AntiSpoofInput],
) -> List[np.ndarray]:
    """GPU work: one forward pass per model for the whole batch."""
    if not inputs:
        return []

    predictor = get_antispoof_model()

    stacked = [
        np.stack([item.crops[i] for item in inputs])
        for i in range(len(predictor.models))
    ]

    summed = predictor.predict_batch(stacked)

    return [summed[i] for i in range(len(inputs))]


def finalize_antispoof(
    item: AntiSpoofInput,
    prediction: np.ndarray,
) -> Result:
    total = float(np.sum(prediction))

    if total <= 0.0:
        return False, 0.0, "no_valid_model_prediction"

    label = int(np.argmax(prediction))
    real_score = float(prediction[REAL_LABEL] / total)
    is_real = label == REAL_LABEL

    if item.is_screen and (not is_real or real_score < 0.85):
        return False, real_score, item.screen_reason

    if is_real and item.is_screen:
        return True, real_score, f"soft_screen_suspect:{item.screen_reason}"

    return is_real, real_score, f"label={label},real_score={real_score:.4f}"


def predict_antispoof_score(
    image_bgr: np.ndarray,
) -> Result:
    """
    Single-image convenience API (kept for liveness_service.py and scripts).
    Returns (is_real, real_score, reason).
    """
    try:
        item, early = prepare_antispoof(image_bgr)

        if early is not None:
            return early

        prediction = infer_antispoof_batch([item])[0]

        return finalize_antispoof(item, prediction)

    except Exception as exc:
        return False, 0.0, f"antispoof_exception:{exc}"