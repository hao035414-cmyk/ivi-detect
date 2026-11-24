"""评测：图级 + 区域级指标，以及代价加权阈值搜索。

真值来源：dataset.py 生成时同步落盘的像素级掩膜（gt/*.png），
按连通域还原为"真值缺陷实例"，与预测框做贪心 IoU 匹配。
匹配准则（任一满足即算命中）：
    IoU >= 0.30  或  预测框中心落在真值连通域内
这样对划痕这类细长目标更宽容：框会偏大，但中心必然落在缺陷上。
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .imageio import imread

from .common import Params


def load_index(root: Path) -> list[dict]:
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def gt_instances(mask: np.ndarray) -> list[np.ndarray]:
    """真值掩膜 → 实例掩膜列表(每项 bool 数组)。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        out.append(labels == i)
    return out


def _iou_box(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def match_counts(pred_boxes, gt_masks) -> tuple[int, int, int]:
    """返回 (TP, FP, FN)。"""
    gt_boxes, gt_centers_ok = [], []
    for m in gt_masks:
        ys, xs = np.where(m)
        gt_boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        gt_centers_ok.append(m)

    used = [False] * len(gt_masks)
    tp = 0
    for pb in pred_boxes:
        best, best_iou = -1, 0.0
        for gi, gb in enumerate(gt_boxes):
            if used[gi]:
                continue
            iou = _iou_box(pb, gb)
            if iou > best_iou:
                best, best_iou = gi, iou
        cx = int((pb[0] + pb[2]) / 2)
        cy = int((pb[1] + pb[3]) / 2)
        if best >= 0 and best_iou >= 0.30:
            used[best] = True
            tp += 1
            continue
        # 回退判据：中心点命中未匹配的真值连通域
        hit = None
        for gi, m in enumerate(gt_centers_ok):
            if used[gi]:
                continue
            if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]:
                hit = gi
                break
        if hit is not None:
            used[hit] = True
            tp += 1
        else:
            tp += 0  # 未命中
    fp = len(pred_boxes) - tp
    fn = sum(1 for u in used if not u)
    return tp, fp, fn


def evaluate(detect_fn, p: Params, root: Path, limit: int | None = None,
             collect: bool = False) -> dict:
    """跑完整个数据集，返回指标字典。"""
    index = load_index(root)
    if limit:
        index = index[:limit]

    tp = fp = fn = 0
    ng_total = ng_hit = 0          # 有缺陷图：是否至少检出一处
    ok_total = ok_fp = 0           # 无缺陷图：是否误报
    times: list[float] = []
    per_type: dict[str, dict] = {}
    failures: list[dict] = []

    for item in index:
        img = imread(str(root / "images" / item["file"]), cv2.IMREAD_GRAYSCALE)
        gt = imread(str(root / "gt" / item["file"]), cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        regions, binary, _ = detect_fn(img, p)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

        pred_boxes = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in regions]
        gts = gt_instances(gt)
        t, f_p, f_n = match_counts(pred_boxes, gts)
        tp += t
        fp += f_p
        fn += f_n

        if item["ok"]:
            ok_total += 1
            if regions:
                ok_fp += 1
        else:
            ng_total += 1
            if regions:
                ng_hit += 1
            if collect and (f_n > 0 or f_p > 0):
                failures.append({
                    "file": item["file"], "gt": len(gts), "pred": len(regions),
                    "fn": f_n, "fp": f_p,
                    "types": [d["type"] for d in item["defects"]],
                    "contrasts": [d.get("contrast") for d in item["defects"]],
                    "areas_px": [d.get("area_px") for d in item["defects"]],
                })
            for d in item["defects"]:
                k = d["type"]
                st = per_type.setdefault(k, {"gt": 0, "tp": 0})
                st["gt"] += 1
                gb = d.get("bbox")
                if not gb:
                    continue
                hit = False
                for pb in pred_boxes:
                    cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                    inside = gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3]
                    if inside or _iou_box(pb, tuple(gb)) >= 0.25:
                        hit = True
                        break
                if hit:
                    st["tp"] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cost = p.fn_cost * fn + p.fp_cost * fp

    result = {
        "images": len(index),
        "ng_images": ng_total,
        "ok_images": ok_total,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "miss_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "ng_image_recall": round(ng_hit / ng_total, 4) if ng_total else 0.0,
        "ok_false_alarm": round(ok_fp / ok_total, 4) if ok_total else 0.0,
        "cost": round(cost, 1),
        "avg_ms": round(float(np.mean(times)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps": round(1000.0 / float(np.mean(times)), 1) if times else 0.0,
    }
    if collect:
        result["failures"] = failures
        result["per_type"] = per_type
    return result


def sweep(detect_fn, base: Params, root: Path, field_name: str,
          values: list, limit: int | None = None) -> list[dict]:
    """在单个参数上扫描，返回每个取值的指标（用于画 trade-off 曲线）。"""
    rows = []
    for v in values:
        p = replace(base, **{field_name: v})
        m = evaluate(detect_fn, p, root, limit=limit)
        m["param"] = v
        rows.append(m)
    return rows
