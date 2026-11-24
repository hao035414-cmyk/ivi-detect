"""图片读写：兼容中文路径。

OpenCV 的 cv2.imread / cv2.imwrite 在 Windows 上遇到非 ASCII 路径会静默失败
（返回 None 或写不出文件），而本项目很可能被放在「答辩材料」这类中文目录下。
这里统一用 np.fromfile + cv2.imdecode 绕开，其余代码一律调用这两个函数。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取图片，支持中文路径；读不到返回 None。"""
    try:
        buf = np.fromfile(str(Path(path)), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, flags)
        return img
    except Exception:
        return None


def imwrite(path, img: np.ndarray) -> bool:
    """写出图片，支持中文路径；成功返回 True。"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception:
        return False
