# face_util/test.py
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import warnings
import time

from face_util.src.anti_spoof_predict import AntiSpoofPredict
from face_util.src.generate_patches import CropImage
from face_util.src.utility import parse_model_name

warnings.filterwarnings("ignore")

# Minivision pretrained weights: 0=real, 1=print, 2=replay
REAL_LABEL = 0


def test(image, model_dir, device_id=0):
    """
    Returns:
      1 = REAL (live)
      0 = SPOOF / failed
    """
    if image is None or getattr(image, "size", 0) == 0:
        return 0

    model_test = AntiSpoofPredict(device_id=device_id, force_cpu=True)
    image_cropper = CropImage()

    h, w = image.shape[:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    try:
        image_bbox = model_test.get_bbox(image)
        if not image_bbox:
            return 0
    except Exception:
        return 0

    prediction = np.zeros((1, 3), dtype=np.float32)

    try:
        model_files = sorted(
            f for f in os.listdir(model_dir)
            if f.lower().endswith(".pth") and os.path.isfile(os.path.join(model_dir, f))
        )
    except Exception:
        return 0

    if not model_files:
        return 0

    for model_name in model_files:
        model_path = os.path.join(model_dir, model_name)

        try:
            h_input, w_input, model_type, scale = parse_model_name(model_name)
        except Exception:
            continue

        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": 1.0 if scale is None else scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        try:
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, model_path)
        except Exception:
            continue

    if float(prediction.sum()) == 0.0:
        return 0

    label = int(np.argmax(prediction))
    return 1 if label == REAL_LABEL else 0