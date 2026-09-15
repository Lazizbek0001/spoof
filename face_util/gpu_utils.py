from __future__ import annotations

import os
from functools import lru_cache


# Must be set BEFORE tensorflow is imported anywhere else.
# Empty string = "use all visible GPUs". "-1" = force CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keep all visible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


@lru_cache(maxsize=1)
def get_runtime_info() -> dict:
    """
    Detect whether TensorFlow sees a GPU. Returns a dict:
      {
        "device": "GPU" | "CPU",
        "gpus": [list of gpu names],
        "tf_version": str,
        "cuda_build": bool,
      }
    """
    info = {
        "device": "CPU",
        "gpus": [],
        "tf_version": "unknown",
        "cuda_build": False,
    }

    try:
        import tensorflow as tf
        info["tf_version"] = tf.__version__
        info["cuda_build"] = bool(tf.test.is_built_with_cuda())

        gpus = tf.config.list_physical_devices("GPU")
        info["gpus"] = [g.name for g in gpus]

        if gpus:
            info["device"] = "GPU"
            # Optional: allow memory growth so TF doesn't grab all VRAM
            try:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        else:
            info["device"] = "CPU"
    except Exception as e:
        info["error"] = str(e)

    return info


def describe_runtime() -> str:
    info = get_runtime_info()
    if info["device"] == "GPU":
        return f"DeepFace runtime: GPU ({', '.join(info['gpus'])}) | TF {info['tf_version']}"
    return f"DeepFace runtime: CPU | TF {info['tf_version']} | CUDA build: {info['cuda_build']}"