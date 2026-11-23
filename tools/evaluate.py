"""一键评测：基线 vs 改进 + 消融 + 阈值代价曲线，输出报告与配图。

用法：
    .venv\\Scripts\\python.exe -m tools.evaluate
产物（out/）：
    eval_report.json    全量指标（供 PPT 引用，避免手抄）
    benchmark.csv       逐项指标表
    tradeoff.csv        判定阈值扫描结果
    fig_compare.png     基线/改进三指标对比
    fig_pertype.png     逐类缺陷召回率对比
    fig_tradeoff.png    漏检-误检代价曲线
    fig_cases.png       典型案例四联图
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import baseline, improved, metrics  # noqa: E402
from core.common import Params, overlay  # noqa: E402
from core.textdraw import filled_label  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

CN_FONT = "C:/Windows/Fonts/msyh.ttc"
TYPE_CN = {"scratch": "划痕", "pit": "凹坑", "stain": "污渍", "oil": "油污"}


def font(size: int):
    try:
        return ImageFont.truetype(CN_FONT, size)
    except Exception:
        return ImageFont.load_default()


def save_bar_chart(path: Path, groups: list[str], series: list[tuple[str, list[float], tuple]],
                   title: str, ymax: float = 1.0, fmt: str = "{:.1%}",
                   size=(900, 520), subtitle: str = ""):
    W, H = size
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title, f_axis, f_val = font(26), font(17), font(15)
    d.text((40, 24), title, font=f_title, fill=(25, 32, 45))
    if subtitle:
        d.text((40, 58), subtitle, font=f_axis, fill=(110, 120, 135))

    L, R, T, B = 90, W - 40, 130, H - 90
    d.line([(L, T), (L, B)], fill=(190, 196, 205), width=2)
    d.line([(L, B), (R, B)], fill=(190, 196, 205), width=2)
    for i in range(5):
        v = ymax * i / 4
        y = B - (B - T) * i / 4
        d.line([(L, y), (R, y)], fill=(232, 236, 242), width=1)
        d.text((L - 62, y - 9), fmt.format(v), font=f_axis, fill=(120, 130, 145))

    n = len(groups)
    gw = (R - L) / n
    bw = min(58, gw / (len(series) + 1.2))
    for gi, g in enumerate(groups):
        gx = L + gw * gi + gw / 2
        for si, (_, vals, color) in enumerate(series):
            v = vals[gi]
            x0 = gx - bw * len(series) / 2 + bw * si
            y0 = B - (B - T) * min(1.0, v / ymax)
            d.rectangle([x0, y0, x0 + bw - 6, B], fill=color)
            d.text((x0 + 2, y0 - 22), fmt.format(v), font=f_val, fill=(50, 58, 70))
        d.text((gx - len(g) * 8, B + 14), g, font=f_axis, fill=(60, 70, 85))

    lx = L
    for name, _, color in series:
        d.rectangle([lx, H - 46, lx + 16, H - 30], fill=color)
        d.text((lx + 22, H - 48), name, font=f_axis, fill=(70, 80, 95))
        lx += 40 + len(name) * 17
    img.save(path)


def save_line_chart(path: Path, xs: list[float], series: list[tuple[str, list[float], tuple]],
                    title: str, xlabel: str, size=(980, 520)):
    W, H = size
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title, f_axis = font(26), font(16)
    d.text((40, 22), title, font=f_title, fill=(25, 32, 45))
    L, R, T, B = 90, W - 150, 100, H - 80
    d.line([(L, T), (L, B)], fill=(190, 196, 205), width=2)
    d.line([(L, B), (R, B)], fill=(190, 196, 205), width=2)

    allv = [v for _, vals, _ in series for v in vals]
    vmax = max(allv) * 1.08
    for i in range(5):
        y = B - (B - T) * i / 4
        d.line([(L, y), (R, y)], fill=(232, 236, 242))
        d.text((L - 56, y - 9), f"{vmax * i / 4:.0f}", font=f_axis, fill=(120, 130, 145))
    for i in range(len(xs)):
        if i % 2:
            continue
        x = L + (R - L) * i / max(1, len(xs) - 1)
        d.text((x - 16, B + 12), f"{xs[i]:.2f}", font=f_axis, fill=(120, 130, 145))
    d.text(((L + R) / 2 - 60, B + 38), xlabel, font=f_axis, fill=(90, 100, 115))

    for name, vals, color in series:
        pts = [(L + (R - L) * i / max(1, len(vals) - 1), B - (B - T) * v / vmax)
               for i, v in enumerate(vals)]
        d.line(pts, fill=color, width=3)
        for p in pts[::2]:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)
    ly = T
    for name, _, color in series:
        d.line([(R + 14, ly + 8), (R + 40, ly + 8)], fill=color, width=2)
        d.text((R + 46, ly), name, font=f_axis, fill=(70, 80, 95))
        ly += 28
    img.save(path)


def main():
    p = Params()
    report: dict = {"dataset": {}, "config": p.__dict__, "models": {}}

    idx = metrics.load_index(DATA)
    report["dataset"] = {
        "images": len(idx),
        "ng": sum(1 for i in idx if not i["ok"]),
        "ok": sum(1 for i in idx if i["ok"]),
        "defect_instances": sum(i["n_defects"] for i in idx),
        "resolution": "640x480",
        "px_per_mm": idx[0]["px_per_mm"],
    }

    # --- 基线：扫 otsu_bias 取最优，保证对照公平
    base_rows = []
    for b in range(-25, 26, 5):
        m = metrics.evaluate(baseline.detect, Params(otsu_bias=b), DATA)
        m["otsu_bias"] = b
        base_rows.append(m)
    best_bias = max(base_rows, key=lambda r: r["f1"])["otsu_bias"]
    best_base = metrics.evaluate(baseline.detect, Params(otsu_bias=best_bias), DATA, collect=True)
    best_base["otsu_bias"] = best_bias
    report["models"]["baseline_tuned"] = best_base
    report["models"]["baseline_sweep"] = base_rows

    # --- 改进方案（默认参数）
    imp = metrics.evaluate(improved.detect, p, DATA, collect=True)
    report["models"]["improved"] = imp

    # --- 消融
    ablation = {
        "full": {k: imp[k] for k in ("precision", "recall", "f1", "miss_rate", "cost")},
        "w/o 光照归一化": {k: v for k, v in
                       metrics.evaluate(improved.detect_no_normalize, p, DATA).items()
                       if k in ("precision", "recall", "f1", "miss_rate", "cost")},
        "w/o 多尺度(单核)": {k: v for k, v in
                        metrics.evaluate(improved.detect_single_scale, p, DATA).items()
                        if k in ("precision", "recall", "f1", "miss_rate", "cost")},
        "baseline(Otsu)": {k: best_base[k] for k in
                           ("precision", "recall", "f1", "miss_rate", "cost")},
    }
    report["ablation"] = ablation

    # --- 阈值扫描（代价曲线）
    sweep_rows = []
    for st in np.round(np.arange(0.30, 0.81, 0.05), 2):
        m = metrics.evaluate(improved.detect, Params(score_thr=float(st)), DATA)
        m["score_thr"] = float(st)
        sweep_rows.append(m)
    report["tradeoff"] = sweep_rows
    cost_best = min(sweep_rows, key=lambda r: r["cost"])
    f1_best = max(sweep_rows, key=lambda r: r["f1"])
    report["operating_points"] = {"cost_optimal": cost_best, "f1_optimal": f1_best}

    (OUT / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- CSV
    with (OUT / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方案", "精确率", "召回率", "F1", "漏检率", "整图误报率",
                    "漏检数", "误检数", "平均耗时ms"])
        for name, m in [("基线 Otsu(调优)", best_base), ("改进方案", imp)]:
            w.writerow([name, m["precision"], m["recall"], m["f1"], m["miss_rate"],
                        m["ok_false_alarm"], m["fn"], m["fp"], m["avg_ms"]])
    with (OUT / "tradeoff.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["score_thr", "精确率", "召回率", "漏检数", "误检数", "代价"])
        for m in sweep_rows:
            w.writerow([m["score_thr"], m["precision"], m["recall"], m["fn"], m["fp"], m["cost"]])

    # --- 图
    save_bar_chart(
        OUT / "fig_compare.png", ["精确率", "召回率", "F1"],
        [("基线 Otsu", [best_base["precision"], best_base["recall"], best_base["f1"]], (120, 140, 168)),
         ("改进方案", [imp["precision"], imp["recall"], imp["f1"]], (32, 120, 220))],
        "基线 vs 改进：整体指标（同数据集、同后处理、均取最优参数）",
        subtitle=f"{report['dataset']['images']} 张图 / {report['dataset']['defect_instances']} 处缺陷 / 640×480")

    types = ["scratch", "pit", "stain", "oil"]
    br = [round(best_base["per_type"].get(t, {"tp": 0})["tp"] /
                max(1, best_base["per_type"].get(t, {"gt": 1})["gt"]), 3) for t in types] \
        if best_base.get("per_type") else [0, 0, 0, 0]
    ir = [round(imp["per_type"].get(t, {"tp": 0})["tp"] /
                max(1, imp["per_type"].get(t, {"gt": 1})["gt"]), 3) for t in types]
    save_bar_chart(
        OUT / "fig_pertype.png", [TYPE_CN[t] for t in types],
        [("基线 Otsu", br, (120, 140, 168)), ("改进方案", ir, (32, 120, 220))],
        "逐类召回率：改进方案在哪类缺陷上获胜、哪类仍是短板",
        subtitle="划痕宽度 1~3 px，与拉丝纹理同频，是当前主要漏检来源")

    save_bar_chart(
        OUT / "fig_ablation.png",
        ["基线 Otsu", "去掉光照归一化", "去掉多尺度", "完整方案"],
        [("F1", [ablation["baseline(Otsu)"]["f1"], ablation["w/o 光照归一化"]["f1"],
                 ablation["w/o 多尺度(单核)"]["f1"], ablation["full"]["f1"]], (32, 120, 220))],
        "消融实验：三处改动的独立贡献", subtitle="每去掉一项，F1 的下降幅度即该项的边际收益")

    save_line_chart(
        OUT / "fig_tradeoff.png",
        [r["score_thr"] for r in sweep_rows],
        [("漏检数 FN", [r["fn"] for r in sweep_rows], (196, 62, 62)),
         ("误检数 FP", [r["fp"] for r in sweep_rows], (60, 140, 90))],
        "判定阈值的代价权衡：漏检与误检不可同时最小", "置信度判定阈值 score_thr")

    # --- 典型案例四联图
    cases = []
    for item in idx:
        if item["ok"] or len(cases) >= 2:
            continue
        if len(item["defects"]) >= 3:
            cases.append(item["file"])
    canvas_rows = []
    for f in cases[:2]:
        img = cv2.imread(str(DATA / "images" / f), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(DATA / "gt" / f), cv2.IMREAD_GRAYSCALE)
        rb, mb, _ = baseline.detect(img, Params(otsu_bias=best_base["otsu_bias"]))
        ri, mi, _ = improved.detect(img, p)
        row = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR),
               overlay(img, rb), overlay(img, ri)]
        labels = ["原图", "真值 GT", f"基线 Otsu（检出 {len(rb)}）", f"改进方案（检出 {len(ri)}）"]
        for r, lab in zip(row, labels):
            r = cv2.resize(r, (420, 315))
            filled_label(r, lab, (6, 6), size=15, fg=(238, 242, 250), bg=(30, 34, 42))
            canvas_rows.append(r)
    if canvas_rows:
        rows = [np.hstack(canvas_rows[i:i + 4]) for i in range(0, len(canvas_rows), 4)]
        cv2.imwrite(str(OUT / "fig_cases.png"), np.vstack(rows))

    print(json.dumps({
        "baseline": {k: best_base[k] for k in ("precision", "recall", "f1", "miss_rate", "avg_ms")},
        "improved": {k: imp[k] for k in ("precision", "recall", "f1", "miss_rate",
                                        "ok_false_alarm", "avg_ms", "fps")},
        "ablation": ablation,
        "cost_optimal": {k: cost_best[k] for k in ("score_thr", "fn", "fp", "cost", "miss_rate")},
        "f1_optimal": {k: f1_best[k] for k in ("score_thr", "fn", "fp", "cost", "miss_rate")},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
