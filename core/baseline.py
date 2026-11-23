"""基线方案：全局 Otsu 阈值 + 形态学 + 面积过滤。

这是产线上最常见的"传统视觉"做法，作为对照基线（B0）。
它的失效机理很明确：
  - 全局单一阈值无法同时适配亮区与暗区 → 光照梯度下暗区漏检
  - 低对比缺陷与材质纹理的灰度重叠 → 阈值抬高则误检、压低则漏检
  - 单尺度结构元无法兼顾细划痕与大片油污
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import Params, Region, analyze_regions


def detect(img: np.ndarray, p: Params) -> tuple[list[Region], np.ndarray, np.ndarray]:
    """返回 (区域列表, 检出掩膜, 对比度图/差分图)。"""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    level = int(np.clip(thr + p.otsu_bias, 1, 254))
    binary = (blur < level).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=1)

    # 对比度图：以全局均值为参考、开运算背景估计
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    diff = np.clip(bg.astype(np.float32) - img.astype(np.float32), 0, 255)

    regions = analyze_regions(binary, diff, p)
    return regions, binary, diff
