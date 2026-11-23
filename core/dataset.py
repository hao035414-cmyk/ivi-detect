"""合成工业表面缺陷数据集生成器.

设计目标：在无真实产线数据的前提下，构建一套"可控、可复现、带像素级真值"的
评测集，用于算法基线对照与失败边界分析。

合成要素（决定难度）：
  1. 光照不均：全局梯度 + 暗角(vignetting)，模拟环形光源偏心
  2. 材质纹理：拉丝/磨砂噪声，抬高背景标准差
  3. 四类缺陷：划痕(scratch) / 凹坑(pit) / 污渍(stain) / 油污(oil)
     - 前两类高对比，后两类低对比（故意制造基线算法的失败场景）
  4. 传感器噪声：高斯噪声 + 轻微散焦

每张图输出 image.png(灰度) 与 gt.png(0/255 真值掩膜)。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMG_W, IMG_H = 640, 480

DEFECT_CLASSES = {
    "scratch": {"label": 1, "cn": "划痕", "severity": 3},
    "pit": {"label": 2, "cn": "凹坑", "severity": 2},
    "stain": {"label": 3, "cn": "污渍", "severity": 1},
    "oil": {"label": 4, "cn": "油污", "severity": 1},
}


def _base_material(rng: np.random.Generator) -> np.ndarray:
    """生成带光照不均与拉丝纹理的金属表面基底。"""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)

    # 全局亮度 + 方向梯度：模拟条形/环形光源偏心
    base = 132.0 + rng.uniform(-18, 18)
    gx, gy = rng.uniform(-0.10, 0.10), rng.uniform(-0.14, 0.14)
    img = base + gx * (xx - IMG_W / 2) + gy * (yy - IMG_H / 2)

    # 暗角
    cx, cy = IMG_W * rng.uniform(0.35, 0.65), IMG_H * rng.uniform(0.35, 0.65)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.65 * np.hypot(IMG_W, IMG_H) * 0.5)
    img -= rng.uniform(10, 30) * r**2

    # 拉丝纹理（水平方向拉伸的各向异性噪声）
    tex = rng.normal(0, 1, (IMG_H, IMG_W)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=9.0, sigmaY=0.8)
    tex /= (tex.std() + 1e-6)
    img += tex * rng.uniform(1.4, 2.6)

    # 大尺度低频起伏（材质不均）
    low = rng.normal(0, 1, (IMG_H // 8, IMG_W // 8)).astype(np.float32)
    low = cv2.resize(low, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
    img += cv2.GaussianBlur(low, (0, 0), 12) * rng.uniform(1.5, 4.0)
    return img


def _draw_defect(img: np.ndarray, mask: np.ndarray, kind: str,
                 rng: np.random.Generator) -> dict:
    """在图上绘制单处缺陷，返回该缺陷元信息。"""
    h, w = IMG_H, IMG_W
    layer = np.zeros((h, w), np.float32)
    info = {"type": kind}

    if kind == "scratch":
        x0, y0 = rng.integers(40, w - 40), rng.integers(40, h - 40)
        ang = rng.uniform(0, np.pi)
        ln = rng.uniform(35, 130)
        x1 = int(x0 + ln * np.cos(ang))
        y1 = int(y0 + ln * np.sin(ang))
        thick = int(rng.integers(1, 4))
        cv2.line(layer, (int(x0), int(y0)), (x1, y1), 1.0, thick, cv2.LINE_AA)
        depth = rng.uniform(-58, -26)          # 比背景暗
        info.update(length_px=round(ln, 1), width_px=thick, contrast=round(-depth, 1))

    elif kind == "pit":
        cx, cy = rng.integers(50, w - 50), rng.integers(50, h - 50)
        ax, ay = rng.integers(3, 9), rng.integers(3, 9)
        cv2.ellipse(layer, (int(cx), int(cy)), (int(ax), int(ay)),
                    rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
        depth = rng.uniform(-70, -34)
        info.update(length_px=float(2 * ax), width_px=float(2 * ay),
                    contrast=round(-depth, 1))

    elif kind == "stain":
        cx, cy = rng.integers(60, w - 60), rng.integers(60, h - 60)
        rad = int(rng.integers(7, 17))
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (int(cx), int(cy)), rad, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), rad / 1.6)
        layer = np.maximum(layer, blob / (blob.max() + 1e-6))
        depth = rng.uniform(-34, -18)          # 低对比
        info.update(length_px=float(2 * rad), width_px=float(2 * rad),
                    contrast=round(-depth, 1))

    else:  # oil
        cx, cy = rng.integers(90, w - 90), rng.integers(90, h - 90)
        rx, ry = rng.integers(18, 40), rng.integers(14, 34)
        blob = np.zeros((h, w), np.float32)
        cv2.ellipse(blob, (int(cx), int(cy)), (int(rx), int(ry)),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), min(rx, ry) / 2.2)
        layer = np.maximum(layer, blob / (blob.max() + 1e-6))
        depth = rng.uniform(-22, -12)           # 极低对比，接近纹理底噪
        info.update(length_px=float(2 * rx), width_px=float(2 * ry),
                    contrast=round(-depth, 1))

    img += layer * depth
    m = layer > 0.25
    mask[m] = 255
    ys, xs = np.where(m)
    info["area_px"] = int(m.sum())
    if len(xs):
        info["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return info


def make_one(seed: int, n_defects: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    img = _base_material(rng)
    mask = np.zeros((IMG_H, IMG_W), np.uint8)
    defects = []

    kinds = list(DEFECT_CLASSES)
    for _ in range(n_defects):
        # 高对比:低对比 = 1:1，保证两类场景都有足够样本
        kind = kinds[rng.integers(0, len(kinds))]
        defects.append(_draw_defect(img, mask, kind, rng))

    # 传感器噪声 + 轻微散焦
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 0.9))
    img += rng.normal(0, rng.uniform(1.5, 3.2), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask, defects


def build(out_dir: Path, n_total: int = 160, start_seed: int = 1000) -> Path:
    """生成数据集；约 25% 为 OK 图（无缺陷）用于统计误检。"""
    img_dir = out_dir / "images"
    gt_dir = out_dir / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    index = []
    rng = np.random.default_rng(20261122)
    for i in range(n_total):
        seed = start_seed + i
        is_ok = rng.random() < 0.25
        n_def = 0 if is_ok else int(rng.integers(1, 5))
        img, mask, defects = make_one(seed, n_def)
        stem = f"{i:04d}_{'OK' if is_ok else 'NG'}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        cv2.imwrite(str(gt_dir / f"{stem}.png"), mask)
        index.append({
            "file": f"{stem}.png",
            "seed": seed,
            "ok": bool(is_ok),
            "n_defects": len(defects),
            "defects": defects,
            "px_per_mm": 12.0,     # 标定：12 px = 1 mm（虚构工件，答辩需声明）
        })

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data"
    build(root)
    print(f"dataset ready -> {root}")
