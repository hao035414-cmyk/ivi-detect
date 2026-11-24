"""评估新增能力：缺陷类型粗分类 + 输入尺度自适应。

用法：python -m tools.eval_extra
产物：out/extra_report.json（供 PPT / 说明书引用）
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import classify, improved, metrics  # noqa: E402
from core.common import Params  # noqa: E402
from core.imageio import imread

DATA = ROOT / "data"
OUT = ROOT / "out"


def collect_samples() -> list[tuple[dict, str, int]]:
    """逐图逐缺陷采集特征；用真值框中心匹配检出区域。"""
    p = Params()
    index = metrics.load_index(DATA)
    rows: list[tuple[dict, str, int]] = []
    for k, item in enumerate(index):
        if item["ok"]:
            continue
        img = imread(str(DATA / "images" / item["file"]), cv2.IMREAD_GRAYSCALE)
        regs, mask, contrast = improved.detect(img, p)
        for d in item["defects"]:
            gb = d.get("bbox")
            if not gb:
                continue
            gx, gy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
            hit = None
            for r in regs:
                if gb[0] <= r.cx <= gb[2] and gb[1] <= r.cy <= gb[3]:
                    hit = r
                    break
            if hit is None:
                continue
            rows.append((classify.region_features(hit, contrast, mask, p.px_per_mm),
                         d["type"], k))
    return rows


def scale_test(scale: float, limit: int = 60) -> dict:
    """把图像与真值按 scale 缩放后评测：naive（原始参数）vs adaptive（尺度归一化）。"""
    p = Params()
    index = metrics.load_index(DATA)[:limit]
    naive_tp = naive_fn = ada_tp = ada_fn = 0

    for item in index:
        img0 = imread(str(DATA / "images" / item["file"]), cv2.IMREAD_GRAYSCALE)
        gt0 = imread(str(DATA / "gt" / item["file"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img0, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        gt = cv2.resize(gt0, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        gts = metrics.gt_instances(gt)
        if not gts:
            continue

        for tag in ("naive", "adaptive"):
            fn = improved.detect if tag == "naive" else improved.detect_adaptive
            # 物理标定随缩放变化：像素/毫米 = 12 * scale
            pp = replace(p, px_per_mm=p.px_per_mm * scale,
                         min_area=max(20, int(round(p.min_area * scale ** 2))))
            regs, _, _ = fn(img, pp)
            boxes = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in regs]
            tp, fp, f_n = metrics.match_counts(boxes, gts)
            if tag == "naive":
                naive_tp += tp
                naive_fn += f_n
            else:
                ada_tp += tp
                ada_fn += f_n

    def rec(tp, fn):
        return round(tp / max(1, tp + fn), 4)

    return {
        "scale": scale,
        "naive_recall": rec(naive_tp, naive_fn),
        "adaptive_recall": rec(ada_tp, ada_fn),
        "naive_miss": naive_fn,
        "adaptive_miss": ada_fn,
    }


def main() -> None:
    rows = collect_samples()
    train = [(f, t) for f, t, k in rows if k % 2 == 0]
    test = [(f, t) for f, t, k in rows if k % 2 == 1]

    rule = classify.search_rule(train)
    train_acc = classify.accuracy(train, rule)
    test_acc = classify.accuracy(test, rule)
    all_acc = classify.accuracy([(f, t) for f, t, _ in rows], rule)

    scales = [scale_test(s, limit=60) for s in (0.5, 1.0, 2.0)]

    report = {
        "samples": len(rows),
        "rule": rule,
        "train": train_acc,
        "test": test_acc,
        "all": all_acc,
        "scale_robustness": scales,
    }
    (OUT / "extra_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
