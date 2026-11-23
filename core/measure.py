"""尺寸测量模块：亚像素边缘 + 稳健直线拟合。

与"框选数像素"相比的两处差别：
  1. 边缘定位用梯度峰值 + 三点抛物线拟合，精度到 0.1 px 级（整数像素边界的 ~1/10）；
  2. 每行/每列各取一个边缘点后做中位数 + MAD 剔除，抑制毛刺、反光点造成的单点跳变，
     并输出重复性(标准差)，用于判断该测量是否可信 —— 这是产线上"敢不敢用"的关键。
"""
from __future__ import annotations

import cv2
import numpy as np


def _subpixel_peak(profile: np.ndarray, idx: int) -> float:
    """对 |梯度| 在 idx 处做三点抛物线拟合，返回亚像素位置。"""
    if idx <= 0 or idx >= len(profile) - 1:
        return float(idx)
    y0, y1, y2 = profile[idx - 1], profile[idx], profile[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(idx)
    delta = 0.5 * (y0 - y2) / denom
    return float(idx + np.clip(delta, -1.0, 1.0))


def _edges_1d(row: np.ndarray) -> tuple[float | None, float | None]:
    """一行像素 → (左边缘亚像素坐标, 右边缘亚像素坐标)。"""
    g = np.abs(np.diff(row.astype(np.float32)))
    if g.size < 8:
        return None, None
    g = cv2.GaussianBlur(g.reshape(1, -1), (0, 0), 1.2).ravel()
    half = len(g) // 2
    li = int(np.argmax(g[:half]))
    ri = int(np.argmax(g[half:])) + half
    # 梯度需显著高于噪声
    if g[li] < max(6.0, g.mean() * 2.5) or g[ri] < max(6.0, g.mean() * 2.5):
        return None, None
    return _subpixel_peak(g, li) + 0.5, _subpixel_peak(g, ri) + 0.5


def _robust(vals: list[float]) -> tuple[float, float, float]:
    """中位数 + MAD 剔除，返回 (中位数, 标准差, 有效样本占比)。"""
    if not vals:
        return 0.0, 0.0, 0.0
    a = np.asarray(vals, np.float32)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    keep = np.abs(a - med) <= max(2.5 * mad, 1.0)
    b = a[keep] if keep.sum() >= max(3, len(a) * 0.3) else a
    return float(np.median(b)), float(b.std()), float(keep.sum() / len(a))


def _part_angle_deg(sub: np.ndarray) -> float:
    """用最小外接矩形估计工件倾角（粗定位，角度精度远高于逐行扫描）。"""
    blur = cv2.GaussianBlur(sub, (5, 5), 0)
    thr, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return 0.0
    (_, _), (_, _), ang = cv2.minAreaRect(c)
    ang = (ang + 90) % 90
    if ang > 45:
        ang -= 90
    return float(ang)


def measure(img: np.ndarray, roi: tuple[int, int, int, int] | None = None,
            px_per_mm: float = 12.0) -> dict:
    """测量 ROI 内工件的宽/高/倾角，返回像素与毫米双单位结果。"""
    x0, y0, x1, y1 = roi if roi else (0, 0, img.shape[1], img.shape[0])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    sub = img[y0:y1, x0:x1]

    lefts, rights, tops, bots = [], [], [], []
    for r in range(sub.shape[0]):
        l, rt = _edges_1d(sub[r])
        if l is not None:
            lefts.append(l)
            rights.append(rt)
    for c in range(sub.shape[1]):
        t, b = _edges_1d(sub[:, c])
        if t is not None:
            tops.append(t)
            bots.append(b)

    w_med, w_std, w_ratio = _robust([r - l for l, r in zip(lefts, rights)])
    h_med, h_std, h_ratio = _robust([b - t for t, b in zip(tops, bots)])

    # 倾角由最小外接矩形给出；逐行扫得的弦长按 cos(θ) 校正为垂直轴长度
    angle = _part_angle_deg(sub)
    corr = float(np.cos(np.deg2rad(angle)))
    w_med, h_med = w_med * corr, h_med * corr

    ok = (w_ratio >= 0.4 and h_ratio >= 0.4 and w_med > 5 and h_med > 5)
    return {
        "roi": (x0, y0, x1, y1),
        "w_px": round(w_med, 2), "h_px": round(h_med, 2),
        "w_mm": round(w_med / px_per_mm, 3), "h_mm": round(h_med / px_per_mm, 3),
        "repeatability_um": round(max(w_std, h_std) / px_per_mm * 1000, 1),
        "angle_deg": round(angle, 3),
        "valid_ratio": (round(w_ratio, 3), round(h_ratio, 3)),
        "confident": bool(ok),
        "px_per_mm": px_per_mm,
    }


def draw_measure(img: np.ndarray, res: dict) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = res["roi"]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 180, 255), 1)
    ay = y0 - 16 if y0 > 34 else y1 + 16
    cv2.arrowedLine(vis, (x0, ay), (x1, ay), (0, 200, 255), 2, tipLength=0.02)
    cv2.arrowedLine(vis, (x1 + 16, y0), (x1 + 16, y1), (0, 200, 255), 2, tipLength=0.03)
    cv2.putText(vis, f"W {res['w_mm']:.2f}mm", (x0, ay - 6 if ay < y0 else ay + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"H {res['h_mm']:.2f}", (min(x1 + 20, vis.shape[1] - 76), (y0 + y1) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"1sigma {res['repeatability_um']:.1f}um",
                (x0, min(y1 + 20, vis.shape[0] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 120), 1, cv2.LINE_AA)
    return vis
