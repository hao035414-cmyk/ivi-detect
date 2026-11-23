"""字符识别模块（喷码/批号 OCR）。

不依赖深度学习与网络：投影法切分 + 形态归一化 + 模板匹配。
适用边界（答辩须讲清）：水平排版、字符等宽、对比度足够、无严重畸变的喷码。
复杂场景（曲面、倾斜、点阵字）应替换为 CRNN/PP-OCR，本模块只作为产线快检的轻量方案。
"""
from __future__ import annotations

import cv2
import numpy as np

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NORM_W, NORM_H = 20, 30


def _make_templates() -> dict[str, np.ndarray]:
    """用系统字体栅格化字符集，作为模板库。"""
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]
    tpl: dict[str, np.ndarray] = {}
    for ch in CHARSET:
        acc = np.zeros((NORM_H, NORM_W), np.float32)
        for f in fonts:
            canvas = np.zeros((NORM_H + 16, NORM_W + 12), np.uint8)
            cv2.putText(canvas, ch, (4, NORM_H + 4), f, 0.78, 255, 2, cv2.LINE_AA)
            ys, xs = np.where(canvas > 30)
            if len(xs) == 0:
                continue
            crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crop = cv2.resize(crop, (NORM_W, NORM_H), interpolation=cv2.INTER_AREA)
            acc += crop.astype(np.float32)
        tpl[ch] = acc / (acc.max() + 1e-6)
    return tpl


_TEMPLATES = None


def templates() -> dict[str, np.ndarray]:
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = _make_templates()
    return _TEMPLATES


def _binarize(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # 喷码通常比背景暗：小窗口自适应，适应反光差异
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY_INV, 25, 8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return bw


def segment(bw: np.ndarray) -> list[np.ndarray]:
    """垂直投影切分字符，返回字符切片列表。"""
    col = (bw > 0).sum(axis=0)
    if col.max() == 0:
        return []
    thr = max(1, int(col.max() * 0.10))
    on = col > thr
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 3:
                segs.append((start, i))
            start = None
    if start is not None and len(on) - start >= 3:
        segs.append((start, len(on)))

    chars = []
    for x0, x1 in segs:
        sub = bw[:, x0:x1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        sub = sub[rows.min():rows.max() + 1, :]
        chars.append(cv2.resize(sub, (NORM_W, NORM_H), interpolation=cv2.INTER_AREA))
    return chars


_ENGINE = None
_ENGINE_TRIED = False


def _engine():
    """优先用轻量 ONNX OCR（PP-OCR 检测+识别），未安装时回退到模板匹配。"""
    global _ENGINE, _ENGINE_TRIED
    if not _ENGINE_TRIED:
        _ENGINE_TRIED = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ENGINE = RapidOCR()
        except Exception:
            _ENGINE = None
    return _ENGINE


def recognize(img: np.ndarray, roi: tuple | None = None) -> dict:
    """对外统一入口：返回 text / conf / per_char / verdict / engine。"""
    eng = _engine()
    if eng is not None:
        crop = img if roi is None else img[roi[1]:roi[3], roi[0]:roi[2]]
        color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
        res, _ = eng(color)
        if not res:
            return {"text": "", "n_chars": 0, "conf": 0.0, "per_char": [],
                    "binary": _binarize(crop), "engine": "onnx", "verdict": "EMPTY"}
        text = "".join(r[1] for r in res)
        confs = [float(r[2]) for r in res]
        boxes = [r[0] for r in res]
        per = [{"ch": seg[1], "conf": round(float(seg[2]), 3)} for seg in res]
        conf = round(float(np.mean(confs)), 3)
        return {"text": text, "n_chars": len(text), "conf": conf, "per_char": per,
                "boxes": boxes, "binary": _binarize(crop), "engine": "onnx",
                "verdict": "PASS" if conf >= 0.75 and len(text) >= 4 else "REVIEW"}
    return recognize_template(img)


def recognize_template(img: np.ndarray) -> dict:
    """回退方案：投影切分 + 模板匹配（依赖喷码字体接近模板库）。"""
    bw = _binarize(img)
    chars = segment(bw)
    tpl = templates()
    text, confs, per_char = [], [], []
    for c in chars:
        cn = c.astype(np.float32) / 255.0
        best, best_ch = -1.0, "?"
        for ch, t in tpl.items():
            # NCC，对亮度整体偏移不敏感
            a = cn - cn.mean()
            b = t - t.mean()
            d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
            s = float((a * b).sum() / d)
            if s > best:
                best, best_ch = s, ch
        text.append(best_ch)
        confs.append(round(best, 3))
        per_char.append({"ch": best_ch, "conf": round(best, 3)})
    avg = round(float(np.mean(confs)) if confs else 0.0, 3)
    return {
        "text": "".join(text),
        "n_chars": len(chars),
        "conf": avg,
        "per_char": per_char,
        "binary": bw,
        "engine": "template",
        "verdict": "PASS" if avg >= 0.55 and len(chars) >= 4 else "REVIEW",
    }
