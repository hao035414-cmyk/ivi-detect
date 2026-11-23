"""缺陷类型粗分类（新增能力）：在"检出"之上回答"这是什么缺陷"。

思路与检测链路保持一致——用可解释的几何 + 反差特征做决策，不用学习模型：
  划痕：极细长（最小外接矩形长宽比大）
  凹坑：紧凑、反差高
  油污：成片、反差低、面积大
  污渍：其余（面积小、反差中等偏低）

决策阈值由「训练划分」网格搜索确定，报告在「测试划分」上——避免自己给自己打分。
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

TYPE_CN = {"scratch": "划痕", "pit": "凹坑", "stain": "污渍", "oil": "油污"}


def region_features(region, contrast: np.ndarray, mask: np.ndarray,
                    px_per_mm: float) -> dict:
    """从区域与反差图提取可解释特征。"""
    h, w = mask.shape[:2]
    y0, y1 = max(0, region.y), min(h, region.y + region.h)
    x0, x1 = max(0, region.x), min(w, region.x + region.w)
    sub_mask = mask[y0:y1, x0:x1] > 0
    sub_contrast = contrast[y0:y1, x0:x1]

    if sub_mask.sum() == 0:
        return {"rect_ratio": 1.0, "density": 0.0, "contrast": 0.0,
                "area_mm2": 0.0, "fill": 0.0}

    vals = sub_contrast[sub_mask]
    cnts, _ = cv2.findContours(sub_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    rect_ratio, density = 1.0, 0.0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        rw, rh = max(rw, 1e-3), max(rh, 1e-3)
        rect_ratio = max(rw, rh) / min(rw, rh)
        density = float(sub_mask.sum()) / (rw * rh)

    return {
        "rect_ratio": float(rect_ratio),
        "density": float(density),
        "contrast": float(vals.mean()),
        "area_mm2": float(region.area_px) / (px_per_mm ** 2),
        "fill": float(sub_mask.mean()),
    }


def classify(feat: dict, rule: dict | None = None) -> str:
    """决策表（阈值可在训练划分上搜出，见 search_rule）。

    判据顺序即物理含义：
      细长 → 划痕；高反差且紧凑 → 凹坑；成片或极低反差 → 油污；其余 → 污渍。
    """
    r = rule or DEFAULT_RULE
    if feat["rect_ratio"] >= r["t_ratio"]:
        return "scratch"
    if feat["contrast"] >= r["t_contrast_hi"]:
        return "pit"
    if feat["area_mm2"] >= r["t_area_oil"] or feat["contrast"] <= r["t_contrast_lo"]:
        return "oil"
    return "stain"


DEFAULT_RULE = {"t_ratio": 3.2, "t_contrast_hi": 40.0,
                "t_area_oil": 7.5, "t_contrast_lo": 18.5}


def search_rule(samples: list[tuple[dict, str]]) -> dict:
    """在给定样本上网格搜索决策表阈值，返回最优规则。"""
    best, best_acc = dict(DEFAULT_RULE), -1.0
    grid = {
        "t_ratio": [2.8, 3.0, 3.2, 3.6, 4.0, 4.4],
        "t_contrast_hi": [34, 37, 40, 43, 46, 49],
        "t_area_oil": [5.0, 6.0, 7.0, 7.5, 8.5, 10.0],
        "t_contrast_lo": [15.0, 17.0, 18.5, 20.0, 21.5],
    }
    for combo in itertools.product(*grid.values()):
        rule = dict(zip(grid.keys(), combo))
        acc = np.mean([classify(f, rule) == gt for f, gt in samples]) if samples else 0
        if acc > best_acc:
            best_acc, best = acc, rule
    return best


def accuracy(samples: list[tuple[dict, str]], rule: dict | None = None) -> dict:
    """整体准确率 + 逐类召回。"""
    if not samples:
        return {"acc": 0.0, "per_type": {}, "n": 0}
    hit, per = 0, {}
    for f, gt in samples:
        p = classify(f, rule)
        per.setdefault(gt, {"gt": 0, "tp": 0})
        per[gt]["gt"] += 1
        if p == gt:
            hit += 1
            per[gt]["tp"] += 1
    return {
        "acc": round(hit / len(samples), 4),
        "n": len(samples),
        "per_type": {k: round(v["tp"] / max(1, v["gt"]), 4) for k, v in per.items()},
    }
