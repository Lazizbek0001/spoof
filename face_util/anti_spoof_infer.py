from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np

from .src.anti_spoof_predict import AntiSpoofPredict
from .src.generate_patches import CropImage
from .src.utility import parse_model_name


MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "resources",
    "anti_spoof_models",
)


# ---------------------------------------------------------------------------
# MiniVision / Silent-Face-Anti-Spoofing class mapping
#
# IMPORTANT:
# Original MiniVision pretrained anti-spoof models use:
#
#     class 1 = REAL / LIVE
#
# Other classes are spoof attacks.
#
# Do NOT use REAL_LABEL = 0 for the original pretrained MiniVision weights.
# ---------------------------------------------------------------------------
REAL_LABEL = 1


# ---------------------------------------------------------------------------
# Cached model objects
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_antispoof_model() -> AntiSpoofPredict:
    """
    Automatically use GPU when CUDA is available.
    If GPU is not available, fall back to CPU.

    The selected device is printed once because this function
    is cached with @lru_cache(maxsize=1).
    """

    runtime = get_antispoof_runtime_info()

    if runtime["device"] == "GPU":
        print("=" * 60)
        print("[Anti-Spoof] GPU FOUND")
        print(f"[Anti-Spoof] GPU name: {runtime['gpu_name']}")
        print(f"[Anti-Spoof] Device: cuda:{runtime['device_id']}")
        print(f"[Anti-Spoof] PyTorch: {runtime['torch_version']}")
        print(f"[Anti-Spoof] CUDA: {runtime['cuda_version']}")
        print("=" * 60)

        device_id = runtime["device_id"]

    else:
        print("=" * 60)
        print("[Anti-Spoof] GPU NOT FOUND")
        print("[Anti-Spoof] Falling back to CPU")
        print(f"[Anti-Spoof] PyTorch: {runtime['torch_version']}")

        if runtime.get("error"):
            print(f"[Anti-Spoof] Error: {runtime['error']}")

        print("=" * 60)

        device_id = 0

    # AntiSpoofPredict should internally use:
    #
    #   cuda:<device_id>
    #
    # when torch.cuda.is_available() is True,
    # otherwise it will use CPU.
    predictor = AntiSpoofPredict(
        device_id=device_id,
    )

    # Print the actual device selected by AntiSpoofPredict,
    # if the class exposes a .device attribute.
    actual_device = getattr(
        predictor,
        "device",
        None,
    )

    if actual_device is not None:
        print(
            f"[Anti-Spoof] Actual inference device: {actual_device}"
        )

    return predictor


@lru_cache(maxsize=1)
def get_cropper() -> CropImage:
    """
    Create CropImage only once.
    """
    return CropImage()


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------

def _iter_model_files(model_dir: str) -> list[str]:
    """
    Return all .pth anti-spoof model files sorted by filename.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Anti-spoof model dir not found: {model_dir}"
        )

    files: list[str] = []

    for name in os.listdir(model_dir):
        path = os.path.join(model_dir, name)

        if (
            os.path.isfile(path)
            and name.lower().endswith(".pth")
        ):
            files.append(path)

    if not files:
        raise FileNotFoundError(
            f"No anti-spoof model files found in: {model_dir}"
        )

    return sorted(files)


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
# Main anti-spoof inference
# ---------------------------------------------------------------------------

def predict_antispoof_score(
    image_bgr: np.ndarray,
) -> Tuple[bool, float, str]:
    """
    Run all available MiniVision anti-spoof models on one image.

    Returns:
        (
            is_real,
            real_score,
            reason,
        )

    Example:

        (
            True,
            0.9821,
            "label=1,real_score=0.9821"
        )

    For the original MiniVision pretrained weights:

        label == 1  -> REAL
        otherwise   -> SPOOF
    """

    if (
        image_bgr is None
        or image_bgr.size == 0
    ):
        return (
            False,
            0.0,
            "empty_image",
        )

    try:
        # ---------------------------------------------------------------
        # Resize input if necessary
        # ---------------------------------------------------------------

        image_bgr = _safe_resize_if_needed(
            image_bgr
        )

        # ---------------------------------------------------------------
        # Additional replay-screen heuristic
        # ---------------------------------------------------------------

        is_screen, screen_reason = (
            _looks_like_screen_replay(
                image_bgr
            )
        )

        # ---------------------------------------------------------------
        # Get cached detector / cropper
        # ---------------------------------------------------------------

        model_test = get_antispoof_model()
        image_cropper = get_cropper()

        # MiniVision outputs three classes.
        prediction = np.zeros(
            (1, 3),
            dtype=np.float32,
        )

        # ---------------------------------------------------------------
        # Detect face bounding box
        # ---------------------------------------------------------------

        try:
            image_bbox = model_test.get_bbox(
                image_bgr
            )

            if not image_bbox:
                return (
                    False,
                    0.0,
                    "no_face_bbox",
                )

        except Exception as exc:
            return (
                False,
                0.0,
                f"bbox_error:{exc}",
            )

        # ---------------------------------------------------------------
        # Run all .pth models
        # ---------------------------------------------------------------

        valid_models = 0

        for model_path in _iter_model_files(
            MODEL_DIR
        ):
            model_name = os.path.basename(
                model_path
            )

            try:
                (
                    h_input,
                    w_input,
                    model_type,
                    scale,
                ) = parse_model_name(
                    model_name
                )

            except Exception:
                continue

            params = {
                "org_img": image_bgr,
                "bbox": image_bbox,
                "scale": (
                    1.0
                    if scale is None
                    else scale
                ),
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            if scale is None:
                params["crop"] = False

            try:
                cropped = image_cropper.crop(
                    **params
                )

                pred = model_test.predict(
                    cropped,
                    model_path,
                )

                prediction += pred
                valid_models += 1

            except Exception:
                continue

        # ---------------------------------------------------------------
        # Validate prediction
        # ---------------------------------------------------------------

        prediction_sum = float(
            prediction.sum()
        )

        if (
            valid_models == 0
            or prediction_sum <= 0.0
        ):
            return (
                False,
                0.0,
                "no_valid_model_prediction",
            )

        # ---------------------------------------------------------------
        # Determine predicted class
        # ---------------------------------------------------------------

        label = int(
            np.argmax(prediction)
        )

        total = float(
            np.sum(prediction)
        )

        # IMPORTANT:
        # REAL_LABEL = 1 for the original MiniVision weights.
        real_score = (
            float(
                prediction[0][REAL_LABEL]
                / total
            )
            if total > 0
            else 0.0
        )

        is_real = (
            label == REAL_LABEL
        )

        # ---------------------------------------------------------------
        # Combine model result with screen heuristic
        # ---------------------------------------------------------------

        if (
            is_screen
            and (
                not is_real
                or real_score < 0.85
            )
        ):
            return (
                False,
                real_score,
                screen_reason,
            )

        # ---------------------------------------------------------------
        # REAL
        # ---------------------------------------------------------------

        if is_real:
            if is_screen:
                return (
                    True,
                    real_score,
                    (
                        "soft_screen_suspect:"
                        f"{screen_reason}"
                    ),
                )

            return (
                True,
                real_score,
                (
                    f"label={label},"
                    f"real_score={real_score:.4f}"
                ),
            )

        # ---------------------------------------------------------------
        # SPOOF
        # ---------------------------------------------------------------

        return (
            False,
            real_score,
            (
                f"label={label},"
                f"real_score={real_score:.4f}"
            ),
        )

    except Exception as exc:
        return (
            False,
            0.0,
            f"antispoof_exception:{exc}",
        )