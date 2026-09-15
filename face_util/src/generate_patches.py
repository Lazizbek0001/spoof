# generate_patches.py
# -*- coding: utf-8 -*-
"""
Create patch from original input image by using bbox coordinate.
Safer version: bbox clipping + invalid-size guards.
"""

import cv2
import numpy as np
from typing import Tuple


def _clip_bbox_xywh(bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x, y, bw, bh = bbox
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    bw = max(1, int(bw))
    bh = max(1, int(bh))
    # Ensure x+bw, y+bh stay inside
    if x + bw > img_w:
        bw = max(1, img_w - x)
    if y + bh > img_h:
        bh = max(1, img_h - y)
    return x, y, bw, bh


class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox

        # protect from zero division and insane scales
        box_w = max(1, int(box_w))
        box_h = max(1, int(box_h))
        scale = float(scale)

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        # shift into bounds
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        # final clamp
        left_top_x = max(0, min(int(left_top_x), src_w - 1))
        left_top_y = max(0, min(int(left_top_y), src_h - 1))
        right_bottom_x = max(left_top_x + 1, min(int(right_bottom_x), src_w - 1))
        right_bottom_y = max(left_top_y + 1, min(int(right_bottom_y), src_h - 1))

        return left_top_x, left_top_y, right_bottom_x, right_bottom_y

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        if org_img is None:
            raise ValueError("org_img is None")

        src_h, src_w = org_img.shape[:2]
        bbox = _clip_bbox_xywh(bbox, src_w, src_h)

        if not crop:
            return cv2.resize(org_img, (int(out_w), int(out_h)))

        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

        patch = org_img[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]
        if patch.size == 0:
            # fallback: just resize original
            patch = org_img

        return cv2.resize(patch, (int(out_w), int(out_h)))
