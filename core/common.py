"""公共数据结构与区域分析工具。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

LEVEL_CN = {1: "轻微", 2: "中等", 3: "严重"}


@dataclass
class Params:
    """检测参数（GUI 可实时调节）。"""
    min_area: int = 80          # 最小缺陷面积(px)
    score_thr: float = 0.58     # 置信度阈值（改进算法/基线共用）
    otsu_bias: int = 0          # 基线算法 Otsu 阈值偏移
    lo_ksize: int = 81          # 背景估计中值核
    k_h: float = 2.5            # 高频通道阈值 = mean + k*MADσ
    sigma_l: float = 2.5        # 低频通道低通 σ
    k_l: float = 3.0            # 低频通道阈值 = -k*MADσ
    px_per_mm: float = 12.0     # 标定：12 px = 1 mm
    fn_cost: float = 10.0       # 漏检代价权重
    fp_cost: float = 1.0        # 误检代价权重


@dataclass
class Region:
    """单个检出区域。"""
    rid: int
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    area_px: int
    contrast: float
    solidity: float
    score: float
    level: int
    px_per_mm: float = 12.0
    extra: dict = field(default_factory=dict)

    @property
    def area_mm2(self) -> float:
        return self.area_px / (self.px_per_mm ** 2)

    @property
    def length_mm(self) -> float:
        return max(self.w, self.h) / self.px_per_mm

    @property
    def level_cn(self) -> str:
        return LEVEL_CN.get(self.level, "-")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        d.update(area_mm2=round(self.area_mm2, 3),
                 length_mm=round(self.length_mm, 2),
                 level_cn=self.level_cn)
        return d


def grade(area_px: int, contrast: float, px_per_mm: float) -> int:
    """缺陷分级：综合面积与对比度，映射产线处置策略。

    严重(3): 面积 ≥ 1.0 mm² 或 对比度 ≥ 40  → 直接判废
    中等(2): 面积 ≥ 0.35 mm² 或 对比度 ≥ 22 → 人工复检
    轻微(1): 其余                            → 放行并记录
    """
    area_mm2 = area_px / (px_per_mm ** 2)
    lv = 3 if area_mm2 >= 3.0 else (2 if area_mm2 >= 0.8 else 1)
    if contrast >= 55 and lv < 3:      # 极小但反差极高（深划痕/崩口）上调一级
        lv += 1
    return lv


def analyze_regions(binary: np.ndarray, diff: np.ndarray,
                    p: Params, weights=(0.5, 0.3, 0.2)) -> list[Region]:
    """对二值掩膜做连通域分析，输出带置信度评分的区域列表。"""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    regions: list[Region] = []
    # 用于归一化评分的参考尺度（鲁棒，避免单张图自归一化导致阈值漂移）
    contrasts, areas = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.min_area:
            continue
        m = labels == i
        contrasts.append(float(diff[m].mean()))
        areas.append(area)
    if not contrasts:
        return regions

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.min_area:
            continue
        m = labels == i
        contrast = float(diff[m].mean())
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea) if cnts else None
        hull_area = cv2.contourArea(cv2.convexHull(cnt)) if cnt is not None else area
        solidity = float(area / hull_area) if hull_area > 1 else 0.0
        # 细长物（划痕）solidity 天然偏低，用长宽比补偿形状分
        aspect = max(w, h) / max(1, min(w, h))
        shape_score = max(solidity, min(1.0, aspect / 3.0))

        c_norm = np.clip(contrast / 45.0, 0, 1)
        a_norm = np.clip(np.log1p(area) / np.log1p(4000), 0, 1)
        score = weights[0] * c_norm + weights[1] * a_norm + weights[2] * shape_score

        regions.append(Region(
            rid=len(regions) + 1, x=x, y=y, w=w, h=h, cx=float(cents[i][0]),
            cy=float(cents[i][1]), area_px=area, contrast=round(contrast, 2),
            solidity=round(solidity, 3), score=round(float(score), 3),
            level=grade(area, contrast, p.px_per_mm), px_per_mm=p.px_per_mm,
            extra={"aspect": round(float(aspect), 2)},
        ))
    return regions


def overlay(img: np.ndarray, regions: list[Region], show_score: bool = True) -> np.ndarray:
    """在原图上绘制检出框与标注（BGR 输出，中文标注走 PIL）。"""
    from .textdraw import filled_label

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    color_by_level = {1: (0, 200, 255), 2: (0, 140, 255), 3: (40, 60, 235)}
    for r in regions:
        c = color_by_level.get(r.level, (0, 255, 0))
        cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), c, 2)
        tag = f"#{r.rid} {r.level_cn}"
        if show_score:
            tag += f" {r.score:.2f}"
        ty = max(2, r.y - 20)
        from .textdraw import text_size
        tw = text_size(tag, 13)[0] + 8
        lx = max(0, min(r.x, vis.shape[1] - tw - 2))
        filled_label(vis, tag, (lx, ty), size=13, fg=c, bg=(24, 28, 36))
    return vis


def mask_vis(binary: np.ndarray) -> np.ndarray:
    """二值掩膜转 BGR 便于显示。"""
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
