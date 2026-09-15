from __future__ import annotations

import os
from functools import lru_cache


# Do NOT set CUDA_VISIBLE_DEVICES="" here.
# Leave it unset so PyTorch / TensorFlow can discover available GPUs.

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# ---------------------------------------------------------------------------
# PyTorch / MiniFAS anti-spoof
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_antispoof_runtime_info() -> dict:
    info = {
        "device": "CPU",
        "device_id": None,
        "gpu_name": None,
        "torch_version": "unknown",
        "cuda_version": None,
        "cuda_available": False,
        "gpu_count": 0,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())

        if torch.cuda.is_available():
            count = torch.cuda.device_count()

            info["gpu_count"] = count
            info["device"] = "GPU"
            info["device_id"] = 0
            info["gpu_name"] = torch.cuda.get_device_name(0)

        else:
            info["device"] = "CPU"

    except Exception as e:
        info["error"] = str(e)
        info["device"] = "CPU"

    return info


# ---------------------------------------------------------------------------
# TensorFlow / DeepFace
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_runtime_info() -> dict:
    """
    Kept backward-compatible with your face_verify_service.py.

    Returns information about the TensorFlow / DeepFace runtime.
    """

    info = {
        "device": "CPU",
        "gpus": [],
        "gpu_names": [],
        "tf_version": "unknown",
        "cuda_build": False,
    }

    try:
        import tensorflow as tf

        info["tf_version"] = tf.__version__
        info["cuda_build"] = bool(
            tf.test.is_built_with_cuda()
        )

        gpus = tf.config.list_physical_devices("GPU")

        info["gpus"] = [
            gpu.name
            for gpu in gpus
        ]

        gpu_names = []

        for gpu in gpus:
            try:
                details = (
                    tf.config.experimental
                    .get_device_details(gpu)
                )

                gpu_name = details.get(
                    "device_name",
                    gpu.name,
                )

            except Exception:
                gpu_name = gpu.name

            gpu_names.append(gpu_name)

        info["gpu_names"] = gpu_names

        if gpus:
            info["device"] = "GPU"

            # Prevent TensorFlow from immediately reserving
            # almost all GPU memory.
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(
                        gpu,
                        True,
                    )
                except RuntimeError:
                    # TensorFlow GPU was already initialized.
                    pass
                except Exception:
                    pass

        else:
            info["device"] = "CPU"

    except Exception as e:
        info["error"] = str(e)
        info["device"] = "CPU"

    return info


# ---------------------------------------------------------------------------
# Console description
# ---------------------------------------------------------------------------

def describe_runtime() -> str:
    anti = get_antispoof_runtime_info()
    deep = get_runtime_info()

    lines = []

    # PyTorch / MiniFAS
    if anti["device"] == "GPU":
        lines.append(
            "[GPU] Anti-spoof / PyTorch: GPU FOUND"
        )
        lines.append(
            f"      GPU: {anti['gpu_name']}"
        )
        lines.append(
            f"      Device: cuda:{anti['device_id']}"
        )
        lines.append(
            f"      PyTorch: {anti['torch_version']}"
        )
        lines.append(
            f"      CUDA: {anti['cuda_version']}"
        )

    else:
        lines.append(
            "[GPU] Anti-spoof / PyTorch: GPU NOT FOUND -> using CPU"
        )
        lines.append(
            f"      PyTorch: {anti['torch_version']}"
        )

        if anti.get("error"):
            lines.append(
                f"      Error: {anti['error']}"
            )

    # TensorFlow / DeepFace
    if deep["device"] == "GPU":
        gpu_names = (
            ", ".join(deep["gpu_names"])
            if deep["gpu_names"]
            else ", ".join(deep["gpus"])
        )

        lines.append(
            "[GPU] DeepFace / TensorFlow: GPU FOUND"
        )
        lines.append(
            f"      GPU: {gpu_names}"
        )
        lines.append(
            f"      TensorFlow: {deep['tf_version']}"
        )

    else:
        lines.append(
            "[GPU] DeepFace / TensorFlow: GPU NOT FOUND -> using CPU"
        )
        lines.append(
            f"      TensorFlow: {deep['tf_version']}"
        )
        lines.append(
            f"      CUDA build: {deep['cuda_build']}"
        )

        if deep.get("error"):
            lines.append(
                f"      Error: {deep['error']}"
            )

    return "\n".join(lines)