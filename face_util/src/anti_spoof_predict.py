# anti_spoof_predict.py
# -*- coding: utf-8 -*-
# CPU-only, inference-optimized version (model cache + inference_mode)

import os
import math
import traceback
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

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
from face_util.src.data_io import transform as trans
from face_util.src.utility import get_kernel, parse_model_name, clip_bbox_xywh

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


@dataclass(frozen=True)
class _ModelKey:
    model_path: str
    kernel_size: Tuple[int, int]
    model_type: str


class Detection:
    """
    Face detector wrapper (OpenCV DNN) forced to CPU.
    """
    def __init__(self):
        stack = traceback.extract_stack()
        dirname = os.path.dirname(stack[-2].filename)

        caffemodel = os.path.join(
            dirname, "..", "resources", "detection_model", "Widerface-RetinaFace.caffemodel"
        )
        deploy = os.path.join(
            dirname, "..", "resources", "detection_model", "deploy.prototxt"
        )

        if not os.path.isfile(caffemodel):
            raise FileNotFoundError(f"Missing detection model: {caffemodel}")
        if not os.path.isfile(deploy):
            raise FileNotFoundError(f"Missing detection prototxt: {deploy}")

        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.detector_confidence = 0.6

    def get_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns bbox as (x, y, w, h) in original image coords.
        If no reliable face found, returns None.
        """
        h, w = img_bgr.shape[:2]
        aspect_ratio = w / max(h, 1)

        resized = img_bgr
        if w * h >= 192 * 192:
            resized = cv2.resize(
                img_bgr,
                (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))),
                interpolation=cv2.INTER_LINEAR,
            )

        blob = cv2.dnn.blobFromImage(resized, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, "data")
        out = self.detector.forward("detection_out").squeeze()

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
    Inference-only anti-spoof predictor:
      - CPU forced by default
      - Caches loaded .pth models (NO reload per request)
      - Uses torch.inference_mode()
    """
    def __init__(self, device_id: int = 0, force_cpu: bool = True, num_threads: Optional[int] = None):
        super().__init__()

        if num_threads and num_threads > 0:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(max(1, min(4, num_threads // 2)))

        self.device = torch.device("cpu") if force_cpu else torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )

        # Cache transform once (your old code recreated it every predict call)
        self._to_tensor = trans.Compose([trans.ToTensor()])

        # Model cache: key -> model (eval)
        self._models: Dict[_ModelKey, torch.nn.Module] = {}

    def _build_and_load_model(self, model_path: str) -> Tuple[torch.nn.Module, Tuple[int, int]]:
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)

        if model_type not in MODEL_MAPPING:
            raise ValueError(f"Unknown model type '{model_type}' parsed from: {model_name}")

        kernel_size = get_kernel(h_input, w_input)
        key = _ModelKey(model_path=os.path.abspath(model_path), kernel_size=kernel_size, model_type=model_type)

        if key in self._models:
            return self._models[key], kernel_size

        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        # Always load weights to CPU (safe on non-GPU machines)
        state_dict = torch.load(model_path, map_location="cpu")

        # Handle DataParallel prefix "module."
        # (your current file does this too) :contentReference[oaicite:2]{index=2}
        first_key = next(iter(state_dict))
        if first_key.startswith("module."):
            from collections import OrderedDict
            new_sd = OrderedDict((k[7:], v) for k, v in state_dict.items())
            model.load_state_dict(new_sd, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

        model.eval()
        self._models[key] = model
        return model, kernel_size

    def predict(self, img_bgr: np.ndarray, model_path: str) -> np.ndarray:
        """
        Returns probabilities shape (1, 3) as numpy float32.
        """
        model, _ = self._build_and_load_model(model_path)

        # BGR -> tensor; keep same behavior as your old code :contentReference[oaicite:3]{index=3}
        x = self._to_tensor(img_bgr).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

        return probs
