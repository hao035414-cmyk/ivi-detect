"""生成演示用样例：喷码识别 3 张（优/一般/差）、尺寸测量 2 张、缺陷检测样例索引。

产物 out/demo/，供界面演示与 PPT 截图使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dataset import make_one  # noqa: E402

OUT = ROOT / "out" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 640, 480


def metal_bg(seed: int, base: float = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = base + 0.05 * (xx - W / 2) - 0.03 * (yy - H / 2)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 9, 0.8)
    img += tex / (tex.std() + 1e-6) * 2.0
    img += rng.normal(0, 2.0, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8)


def with_text(bg: np.ndarray, text: str, scale=1.0, blur=0.0,
              contrast=70, seed=0) -> np.ndarray:
    pil = Image.fromarray(bg).convert("L")
    d = ImageDraw.Draw(pil)
    font = None
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/consolab.ttf",
              "C:/Windows/Fonts/msyhbd.ttc"):
        if Path(p).exists():
            font = ImageFont.truetype(p, int(58 * scale))
            break
    if font is None:
        font = ImageFont.load_default()
    txt = Image.new("L", (W, H), 0)
    dt = ImageDraw.Draw(txt)
    bbox = dt.textbbox((0, 0), text, font=font)
    x = (W - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (H - (bbox[3] - bbox[1])) // 2 - bbox[1]
    dt.text((x, y), text, font=font, fill=255)
    t = np.array(txt, np.float32) / 255.0
    out = bg.astype(np.float32) - t * contrast
    rng = np.random.default_rng(seed)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out += rng.normal(0, 2.5, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_measure(seed: int, angle: float, w=260, h=170) -> np.ndarray:
    """在暗背景上放一块亮色工件，带轻微光照与噪声，用于尺寸测量演示。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = 46 + 0.02 * (xx - W / 2) + rng.normal(0, 3.0, (H, W))
    canvas = np.zeros((H, W), np.uint8)
    rect = ((W // 2, H // 2), (w, h), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(canvas, [box], 255)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.1)
    edge = canvas.astype(np.float32) / 255.0
    img = img + edge * 135
    # 工件内部材质纹理
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 6, 0.6)
    img += tex / (tex.std() + 1e-6) * 2.2 * edge
    img += rng.normal(0, 2.0, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    # --- 喷码
    cv2.imwrite(str(OUT / "ocr_good.png"), with_text(metal_bg(11), "SN2026A1", 0.95, 0.4, 82, 1))
    cv2.imwrite(str(OUT / "ocr_normal.png"), with_text(metal_bg(12, 118), "LOT8842B", 0.85, 0.9, 52, 2))
    cv2.imwrite(str(OUT / "ocr_hard.png"), with_text(metal_bg(13, 136), "QC3971X", 0.72, 1.6, 30, 3))

    # --- 尺寸
    cv2.imwrite(str(OUT / "measure_0.png"), make_measure(21, 0.0, 260, 170))
    cv2.imwrite(str(OUT / "measure_8.png"), make_measure(22, 8.0, 300, 190))

    # --- 缺陷样例（挑一张中等难度）
    for seed, name in ((20261131, "defect_medium.png"), (20261145, "defect_hard.png")):
        img, mask, _ = make_one(seed, 3)
        cv2.imwrite(str(OUT / name), img)
        cv2.imwrite(str(OUT / name.replace(".png", "_gt.png")), mask)

    print("demo ->", OUT)


if __name__ == "__main__":
    main()
