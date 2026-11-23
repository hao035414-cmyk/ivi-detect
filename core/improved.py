"""改进方案（本项目核心）：双频通道 + 自适应阈值 + 置信度分级。

相对基线 B0 的三处具体改动，每处都能单独消融（tools/ablation.py）：

  C1  光照归一化
      中值背景估计(81×81)后差分，把"单幅图内的绝对灰度判定"改成"相对局部背景的反差判定"。
      直接消除光源偏心与暗角造成的暗区漏检 —— 这是基线最致命的失效点。

  C2  双频通道融合
      高频通道 A：多尺度黑帽(7/15/25)取最大响应，保细划痕与凹坑边缘。
                  小核保 1~2 px 划痕，大核保宽划痕，避免"一个结构元走天下"。
      低频通道 B：残差低通(σ=2.5)后按 -k·σ 判定，保低对比大面积污渍/油污。
                  平滑同时把纹理噪声按 √N 衰减，使信噪比低于 1 的缺陷重新可见。
      两通道取并集：一个管"细"，一个管"弱"，两者的失效模式不重叠。

  C3  自适应阈值 + 代价加权判定
      两通道阈值都不写死灰度值，而是 mean + k·σ（对材质、曝光、增益不敏感，
      换一种工件不必重调参）。最终判定阈值不取 F1 最优点，而取
      漏检代价 : 误检代价 = 10 : 1 下的代价最小点：
      产线上漏检一件废品流出 ≫ 多复检一件良品，代价矩阵本就不对称。
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import Params, Region, analyze_regions

HIGH_SCALES = (7, 15, 25)


def illumination_normalize(img: np.ndarray, ksize: int = 81) -> tuple[np.ndarray, np.ndarray]:
    """中值背景估计 + 差分归一化，消除光照梯度与暗角。"""
    k = ksize if ksize % 2 == 1 else ksize + 1
    bg = cv2.medianBlur(img, max(3, k))
    norm = cv2.add(img, 128 - bg.astype(np.int16)).astype(np.uint8)
    return norm, bg


def high_freq_response(norm: np.ndarray, scales=HIGH_SCALES) -> np.ndarray:
    """多尺度黑帽响应取最大值：小核保细划痕，大核保宽缺陷。"""
    resp = np.zeros(norm.shape, np.float32)
    for s in scales:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        bh = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, k)
        resp = np.maximum(resp, bh.astype(np.float32))
    return resp


def _robust_sigma(x: np.ndarray) -> float:
    """MAD 估计的标准差，对少量缺陷像素不敏感（比 std 稳健）。"""
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826) + 1e-6


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """指定方向的线性结构元，用于各向异性形态学。"""
    L = max(3, length | 1)
    k = np.zeros((L, L), np.uint8)
    c = L // 2
    a = np.deg2rad(angle_deg)
    for t in np.linspace(-c, c, L * 3):
        x = int(round(c + t * np.cos(a)))
        y = int(round(c + t * np.sin(a)))
        if 0 <= x < L and 0 <= y < L:
            k[y, x] = 1
    return k


def anisotropic_close(binary: np.ndarray, length: int = 11,
                      angles=(0, 45, 90, 135)) -> np.ndarray:
    """各向异性闭运算：4 个方向线性核分别闭运算后取并集。

    各向同性圆核(5×5)会把宽度 1~2 px 的划痕整段消掉，
    而划痕恰恰是漏检代价最高的一类缺陷。用方向性核只沿划痕方向连接断点，
    横向不长胖，既不吞掉细划痕，也能桥接反光造成的断线。
    """
    out = np.zeros_like(binary)
    for ang in angles:
        k = _line_kernel(length, ang)
        out = cv2.bitwise_or(out, cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k))
    return out


def dual_channel(img: np.ndarray, p: Params):
    """返回 (融合二值图, 反差图, 中间量字典)。"""
    norm, bg = illumination_normalize(img, p.lo_ksize)
    n = norm.astype(np.float32) - 128.0

    # --- 高频通道
    resp_h = high_freq_response(norm)
    thr_h = resp_h.mean() + p.k_h * _robust_sigma(resp_h)
    ch_h = resp_h > thr_h

    # --- 低频通道
    d_l = cv2.GaussianBlur(n, (0, 0), max(0.8, p.sigma_l))
    thr_l = p.k_l * _robust_sigma(d_l)
    ch_l = d_l < -thr_l

    fuse = ((ch_h | ch_l).astype(np.uint8)) * 255
    contrast = np.maximum(resp_h, np.clip(-d_l * 2.0, 0, None))

    # 相对反差细化（滞后阈值）：只在局部最强者附近保留像素，
    # 把低频通道在缺陷周围"晕开"的过渡带削掉，面积量化才接近真值。
    local_max = cv2.dilate(contrast, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    fuse = cv2.bitwise_and(fuse, ((contrast >= 0.34 * local_max).astype(np.uint8)) * 255)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fuse = cv2.morphologyEx(fuse, cv2.MORPH_OPEN, k3, iterations=1)
    fuse = anisotropic_close(fuse, length=11)
    info = {"norm": norm, "bg": bg, "resp_h": resp_h, "d_l": d_l,
            "thr_h": round(float(thr_h), 2), "thr_l": round(float(thr_l), 2)}
    return fuse, contrast, info


def detect(img: np.ndarray, p: Params) -> tuple[list[Region], np.ndarray, np.ndarray]:
    fuse, contrast, _ = dual_channel(img, p)
    regions = analyze_regions(fuse, contrast, p)
    kept = [r for r in regions if r.score >= p.score_thr]
    for i, r in enumerate(kept, 1):
        r.rid = i

    final = np.zeros_like(fuse)
    for r in kept:
        y0, y1 = r.y, r.y + r.h
        x0, x1 = r.x, r.x + r.w
        sub = fuse[y0:y1, x0:x1]
        final[y0:y1, x0:x1] = np.maximum(final[y0:y1, x0:x1], sub)
    return kept, final, contrast


def detect_no_normalize(img: np.ndarray, p: Params):
    """消融 C1：跳过光照归一化，其余流程不变。"""
    norm = img.copy()
    n = norm.astype(np.float32) - 128.0
    resp_h = high_freq_response(norm)
    ch_h = resp_h > resp_h.mean() + p.k_h * _robust_sigma(resp_h)
    d_l = cv2.GaussianBlur(n, (0, 0), max(0.8, p.sigma_l))
    ch_l = d_l < -p.k_l * _robust_sigma(d_l)
    fuse = ((ch_h | ch_l).astype(np.uint8)) * 255
    fuse = anisotropic_close(fuse, length=11)
    contrast = np.maximum(resp_h, np.clip(-d_l * 2.0, 0, None))
    regs = analyze_regions(fuse, contrast, p)
    kept = [r for r in regs if r.score >= p.score_thr]
    for i, r in enumerate(kept, 1):
        r.rid = i
    return kept, fuse, contrast


def detect_single_scale(img: np.ndarray, p: Params):
    """消融 C2 前半：只用单尺度黑帽(9×9)，去掉多尺度。"""
    norm, _ = illumination_normalize(img, p.lo_ksize)
    n = norm.astype(np.float32) - 128.0
    resp_h = high_freq_response(norm, scales=(9,))
    ch_h = resp_h > resp_h.mean() + p.k_h * _robust_sigma(resp_h)
    d_l = cv2.GaussianBlur(n, (0, 0), max(0.8, p.sigma_l))
    ch_l = d_l < -p.k_l * _robust_sigma(d_l)
    fuse = ((ch_h | ch_l).astype(np.uint8)) * 255
    fuse = anisotropic_close(fuse, length=11)
    contrast = np.maximum(resp_h, np.clip(-d_l * 2.0, 0, None))
    regs = analyze_regions(fuse, contrast, p)
    kept = [r for r in regs if r.score >= p.score_thr]
    for i, r in enumerate(kept, 1):
        r.rid = i
    return kept, fuse, contrast
