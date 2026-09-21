# anti_spoof_predict.py
# -*- coding: utf-8 -*-
"""
MiniFASNet anti-spoof predictor.

Changes vs. the previous version:
  - GPU is actually used (force_cpu now defaults to False).
  - All .pth models are loaded once (load_dir) instead of being resolved per frame.
  - predict_batch(): one forward pass per model for a whole batch of crops,
    FP16 autocast on CUDA, a single device->host copy at the end.
  - The OpenCV face detector is created per thread (cv2.dnn.Net is not
    thread-safe; sharing one net across worker threads mixes up frames).
"""

from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from face_util.src.model_lib.MiniFASNet import (
    MiniFASNetV1,
    MiniFASNetV2,
    MiniFASNetV1SE,
    MiniFASNetV2SE,
)
from face_util.src.utility import get_kernel, parse_model_name, clip_bbox_xywh

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}

_RESOURCES = Path(__file__).resolve().parent.parent / "resources"


@dataclass(frozen=True)
class LoadedModel:
    path: str
    name: str
    h_input: int
    w_input: int
    scale: Optional[float]
    model: torch.nn.Module


class Detection:
    """
    RetinaFace (Caffe) face detector on OpenCV DNN.

    One cv2.dnn.Net per thread: setInput() + forward() on a shared net is a
    race when several worker threads detect at the same time.
    """

    def __init__(self) -> None:
        self._caffemodel = str(
            _RESOURCES / "detection_model" / "Widerface-RetinaFace.caffemodel"
        )
        self._deploy = str(_RESOURCES / "detection_model" / "deploy.prototxt")

        if not os.path.isfile(self._caffemodel):
            raise FileNotFoundError(f"Missing detection model: {self._caffemodel}")
        if not os.path.isfile(self._deploy):
            raise FileNotFoundError(f"Missing detection prototxt: {self._deploy}")

        self._local = threading.local()
        self.detector_confidence = 0.6

    def _net(self) -> "cv2.dnn.Net":
        net = getattr(self._local, "net", None)
        if net is None:
            net = cv2.dnn.readNetFromCaffe(self._deploy, self._caffemodel)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._local.net = net
        return net

    def get_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns bbox as (x, y, w, h) in original image coords,
        or None when no reliable face is found.
        """
        h, w = img_bgr.shape[:2]
        aspect_ratio = w / max(h, 1)

        resized = img_bgr
        if w * h >= 192 * 192:
            resized = cv2.resize(
                img_bgr,
                (
                    int(192 * math.sqrt(aspect_ratio)),
                    int(192 / math.sqrt(aspect_ratio)),
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        net = self._net()
        blob = cv2.dnn.blobFromImage(resized, 1, mean=(104, 117, 123))
        net.setInput(blob, "data")
        out = net.forward("detection_out").squeeze()

        if out.ndim != 2 or out.shape[1] < 7:
            return None

        conf = out[:, 2]
        best = int(np.argmax(conf))
        if float(conf[best]) < self.detector_confidence:
            return None

        left, top, right, bottom = (
            out[best, 3] * w,
            out[best, 4] * h,
            out[best, 5] * w,
            out[best, 6] * h,
        )
        bbox = (int(left), int(top), int(right - left + 1), int(bottom - top + 1))
        return clip_bbox_xywh(bbox, w, h)


class AntiSpoofPredict(Detection):
    """
    Inference-only anti-spoof predictor.

    Typical use:
        p = AntiSpoofPredict(model_dir=...)          # loads every .pth once
        probs = p.predict_batch([crops_m0, crops_m1])  # (B, 3) summed probs
    """

    def __init__(
        self,
        device_id: int = 0,
        force_cpu: bool = False,
        model_dir: Optional[str] = None,
        use_fp16: bool = True,
        num_threads: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_threads and num_threads > 0:
            torch.set_num_threads(num_threads)

        use_cuda = (not force_cpu) and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
        self.use_fp16 = bool(use_fp16 and use_cuda)

        if use_cuda:
            # TF32 is free speed on Ampere/Ada (RTX 30/40 series).
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.models: List[LoadedModel] = []
        self._by_path: Dict[str, LoadedModel] = {}
        self._load_lock = threading.Lock()

        if model_dir:
            self.load_dir(model_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dir(self, model_dir: str) -> List[LoadedModel]:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Anti-spoof model dir not found: {model_dir}")

        names = sorted(
            n for n in os.listdir(model_dir) if n.lower().endswith(".pth")
        )
        if not names:
            raise FileNotFoundError(f"No anti-spoof model files found in: {model_dir}")

        for name in names:
            try:
                self.load_model(os.path.join(model_dir, name))
            except Exception as exc:
                print(f"[Anti-Spoof] skipping model {name}: {exc}")

        if not self.models:
            raise RuntimeError(f"No usable anti-spoof models in: {model_dir}")

        return self.models

    def load_model(self, model_path: str) -> LoadedModel:
        key = os.path.abspath(model_path)

        with self._load_lock:
            cached = self._by_path.get(key)
            if cached is not None:
                return cached

            name = os.path.basename(model_path)
            h_input, w_input, model_type, scale = parse_model_name(name)

            if model_type not in MODEL_MAPPING:
                raise ValueError(f"Unknown model type '{model_type}' in: {name}")

            model = MODEL_MAPPING[model_type](
                conv6_kernel=get_kernel(h_input, w_input)
            )

            state_dict = torch.load(model_path, map_location="cpu")
            if next(iter(state_dict)).startswith("module."):
                state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
            model.load_state_dict(state_dict, strict=True)

            model.eval().to(self.device)

            loaded = LoadedModel(
                path=key,
                name=name,
                h_input=h_input,
                w_input=w_input,
                scale=scale,
                model=model,
            )
            self._by_path[key] = loaded
            self.models.append(loaded)
            return loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forward(self, model: torch.nn.Module, batch_u8: np.ndarray) -> torch.Tensor:
        """batch_u8: (B, H, W, 3) uint8 BGR -> (B, 3) float32 probs on device."""
        x = torch.from_numpy(np.ascontiguousarray(batch_u8))

        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)

        # Same preprocessing as the original ToTensor(): HWC -> CHW, float,
        # NOT divided by 255.
        x = x.permute(0, 3, 1, 2).float().contiguous()

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
            logits = model(x)

        return F.softmax(logits.float(), dim=1)

    @torch.inference_mode()
    def predict_batch(self, crops_per_model: Sequence[np.ndarray]) -> np.ndarray:
        """
        crops_per_model[i] is a (B, H, W, 3) uint8 array for self.models[i].
        Returns the per-image sum of probabilities over all models: (B, 3).
        """
        if len(crops_per_model) != len(self.models):
            raise ValueError(
                f"expected {len(self.models)} crop batches, "
                f"got {len(crops_per_model)}"
            )

        total: Optional[torch.Tensor] = None
        for loaded, batch in zip(self.models, crops_per_model):
            probs = self._forward(loaded.model, batch)
            total = probs if total is None else total + probs

        return total.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def predict(self, img_bgr: np.ndarray, model_path: str) -> np.ndarray:
        """Legacy single-image API. Returns probabilities of shape (1, 3)."""
        loaded = self.load_model(model_path)
        probs = self._forward(loaded.model, img_bgr[None, ...])
        return probs.cpu().numpy().astype(np.float32)