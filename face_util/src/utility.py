# utility.py
# -*- coding: utf-8 -*-

from datetime import datetime
import os
from typing import Tuple


def get_time():
    return (str(datetime.now())[:-10]).replace(" ", "-").replace(":", "-")


def get_kernel(height: int, width: int) -> Tuple[int, int]:
    return ((int(height) + 15) // 16, (int(width) + 15) // 16)


def get_width_height(patch_info: str) -> Tuple[int, int]:
    # original behavior :contentReference[oaicite:7]{index=7}
    w_input = int(patch_info.split("x")[-1])
    h_input = int(patch_info.split("x")[0].split("_")[-1])
    return w_input, h_input


def parse_model_name(model_name: str):
    """
    Expected formats (common in MiniFASNet anti-spoof repos):
      - "2.7_80x80_MiniFASNetV2.pth"
      - "4_0_0_80x80_MiniFASNetV1SE.pth"
      - "org_80x80_MiniFASNetV2.pth"
    Returns:
      (h_input:int, w_input:int, model_type:str, scale:float|None)
    """
    base = os.path.basename(model_name)

    if not base.endswith(".pth"):
        raise ValueError(f"Model name must end with .pth: {model_name}")

    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected model name format: {model_name}")

    # last part includes model type with .pth
    model_type = base.split(".pth")[0].split("_")[-1]

    # the size token is the one like "80x80"
    size_token = parts[-2]  # matches your old logic :contentReference[oaicite:8]{index=8}
    if "x" not in size_token:
        raise ValueError(f"Missing HxW token in model name: {model_name}")

    h_input_s, w_input_s = size_token.split("x")
    h_input, w_input = int(h_input_s), int(w_input_s)

    scale_token = parts[0]
    scale = None if scale_token == "org" else float(scale_token)

    return h_input, w_input, model_type, scale


def make_if_not_exist(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def clip_bbox_xywh(bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    bbox: (x,y,w,h) -> clipped to image bounds.
    """
    x, y, w, h = bbox
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, int(w))
    h = max(1, int(h))
    if x + w > img_w:
        w = max(1, img_w - x)
    if y + h > img_h:
        h = max(1, img_h - y)
    return x, y, w, h
