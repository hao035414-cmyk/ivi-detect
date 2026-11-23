"""中文文本绘制（OpenCV 的 putText 不支持中文，统一走 PIL）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhl.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
_cache: dict[int, ImageFont.FreeTypeFont] = {}


def get_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _cache:
        return _cache[size]
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                f = ImageFont.truetype(p, size)
                _cache[size] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _cache[size] = f
    return f


def put_text(img_bgr: np.ndarray, text: str, org: tuple[int, int], size: int = 14,
             color: tuple[int, int, int] = (255, 255, 255),
             bg: tuple[int, int, int] | None = (20, 24, 32),
             pad: int = 3) -> np.ndarray:
    """在 BGR 图上绘制中文文本，org 为左上角。原地修改并返回。"""
    pil = Image.fromarray(img_bgr[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    f = get_font(size)
    x, y = org
    if bg is not None:
        bbox = d.textbbox((x, y), text, font=f)
        d.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=bg[::-1])
    d.text((x, y), text, font=f, fill=color[::-1])
    out = np.array(pil)[:, :, ::-1]
    img_bgr[:] = out
    return img_bgr


def text_size(text: str, size: int = 14) -> tuple[int, int]:
    f = get_font(size)
    tmp = Image.new("RGB", (8, 8))
    return ImageDraw.Draw(tmp).textbbox((0, 0), text, font=f)[2:]


def filled_label(img_bgr: np.ndarray, text: str, org: tuple[int, int], size: int = 14,
                 fg=(255, 255, 255), bg=(30, 34, 42)) -> np.ndarray:
    """带实心底衬的标签，用于结果框标注。"""
    return put_text(img_bgr, text, org, size=size, color=fg, bg=bg, pad=3)
