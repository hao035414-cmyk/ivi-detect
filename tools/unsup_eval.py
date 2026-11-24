"""训练无监督异常检测模型（只用良品图）+ 评测对比。

用法：
  python -m tools.unsup_eval            # 训练 + 评测 + 写 out/unsup_report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from core import anomaly, improved, metrics
from core.common import Params

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"


def load_ok_images(index) -> list[np.ndarray]:
    imgs = []
    for item in index:
        if item["ok"]:
            img = cv2.imread(str(DATA / "images" / item["file"]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)
    return imgs


def main() -> None:
    p = Params()
    index = metrics.load_index(DATA)
    ok_imgs = load_ok_images(index)
    print(f"训练图（良品）：{len(ok_imgs)} 张", flush=True)

    t0 = time.perf_counter()
    model = anomaly.train(ok_imgs)
    train_ms = (time.perf_counter() - t0) * 1000
    model.save(str(OUT / "anomaly_model.npz"))
    print(f"训练完成 {train_ms:.0f} ms | bank={model.bank.shape} | thr={model.thr:.4f} | hi={model.score_hi:.4f}",
          flush=True)

    # 单帧耗时
    t0 = time.perf_counter()
    for _ in range(3):
        anomaly.detect_unsup(ok_imgs[0], model, p)
    infer_ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"单帧推理 ≈ {infer_ms:.1f} ms", flush=True)

    # 评测：无监督 / 改进方案 / 两者并集
    def run_unsup(img, params):
        return anomaly.detect_unsup(img, model, params)

    m_unsup = metrics.evaluate(run_unsup, p, DATA)
    m_improved = metrics.evaluate(improved.detect, p, DATA)

    def run_union(img, params):
        r1, mask1, c1 = improved.detect(img, params)
        r2, mask2, c2 = anomaly.detect_unsup(img, model, params)
        merged = cv2.bitwise_or(mask1, mask2)
        combined = np.maximum(c1, c2)
        from core.common import analyze_regions
        return analyze_regions(merged, combined, params), merged, combined

    m_union = metrics.evaluate(run_union, p, DATA)

    report = {
        "train_images": len(ok_imgs),
        "train_ms": round(train_ms, 1),
        "infer_ms": round(infer_ms, 1),
        "bank": list(model.bank.shape),
        "thr": round(model.thr, 4),
        "unsupervised": {k: m_unsup[k] for k in
                         ("precision", "recall", "f1", "miss_rate", "ok_false_alarm", "avg_ms")},
        "improved": {k: m_improved[k] for k in
                     ("precision", "recall", "f1", "miss_rate", "ok_false_alarm", "avg_ms")},
        "union": {k: m_union[k] for k in
                  ("precision", "recall", "f1", "miss_rate", "ok_false_alarm", "avg_ms")},
        "per_type_unsup": m_unsup.get("per_type"),
    }
    (OUT / "unsup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
