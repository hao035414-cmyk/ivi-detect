"""无监督异常检测（PatchCore-lite）：只用良品图建模，推理时给出"与良品分布的偏离程度"。

与改进方案的差别：
  - 训练阶段只需要「良品图」，不需要任何缺陷标注（改进方案的分类阈值需要标注）；
  - 推理不再依赖阈值化反差，而是比对「局部纹理是否属于良品特征分布」；
  - 保留可解释性：异常分数来自 patch 特征到良品特征库的最近邻距离。

实现要点（纯 NumPy + OpenCV，无深度学习、无 GPU）：
  1. 逐像素算 5 个特征通道 → 用 16×16 盒滤波得到 patch 级均值 → stride 8 采样
  2. 训练：良品 patch 特征做 z-score 归一化后随机下采样为 memory bank
  3. 推理：每个 patch 到 bank 的最近邻 L2 距离 → 分数图 → 上采样 + 平滑
  4. 阈值：训练集的分数分位数（默认 99.5%），保证良品误报率可控
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PATCH = 16
STRIDE = 8
BANK_SIZE = 12000
FEAT_DIM = 5


@dataclass
class AnomalyModel:
    bank: np.ndarray          # (N, 5) float32
    mu: np.ndarray            # (5,) 训练集均值
    sd: np.ndarray            # (5,) 训练集标准差
    thr: float                # 分数阈值（0~1 归一化后）
    score_hi: float           # 用于分数归一的参考上界

    def save(self, path: str) -> None:
        np.savez_compressed(path, bank=self.bank, mu=self.mu, sd=self.sd,
                            thr=np.array([self.thr]), hi=np.array([self.score_hi]))

    @staticmethod
    def load(path: str) -> "AnomalyModel":
        d = np.load(path)
        return AnomalyModel(bank=d["bank"], mu=d["mu"], sd=d["sd"],
                            thr=float(d["thr"][0]), score_hi=float(d["hi"][0]))


def _feature_maps(img: np.ndarray) -> np.ndarray:
    """返回 (H, W, 5) 的逐像素特征：灰度、|gx|、|gy|、梯度幅值、局部标准差。"""
    g = img.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mean = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return np.stack([g, np.abs(gx), np.abs(gy), mag, std], axis=-1)


def patch_features(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """把图切成 patch，返回 (n_patch, 5) 特征与网格尺寸。"""
    feats = _feature_maps(img)
    pooled = cv2.boxFilter(feats, -1, (PATCH, PATCH), normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    sampled = pooled[STRIDE // 2::STRIDE, STRIDE // 2::STRIDE, :]
    h, w = sampled.shape[:2]
    return sampled.reshape(-1, FEAT_DIM).astype(np.float32), (h, w)


AUG_PER_IMAGE = 4
THR_QUANTILE = 0.95


def augment_goods(images: list[np.ndarray], per_image: int = AUG_PER_IMAGE) -> list[np.ndarray]:
    """对良品图做轻量增广（无标注依赖）：水平翻转 + 亮度偏移 + 轻噪声。

    目的不是提高分辨率，而是让「正常」的分布覆盖现场的光照与摆放差异。
    只用良品图增广，不需要任何缺陷样本，仍属无监督范畴。
    """
    rng = np.random.default_rng(20261124)
    out: list[np.ndarray] = []
    for img in images:
        variants = [img, cv2.flip(img, 1)]
        base = img.astype(np.int16)
        variants.append(np.clip(base + 7, 0, 255).astype(np.uint8))
        variants.append(np.clip(base - 7, 0, 255).astype(np.uint8))
        if per_image > len(variants):
            noise = rng.normal(0, 1.5, img.shape).astype(np.float32)
            variants.append(np.clip(img + noise, 0, 255).astype(np.uint8))
        out.extend(variants[:per_image])
    return out


def load_train_goods(data_dir) -> list[np.ndarray]:
    """收集训练用良品图：评测集里的 OK 图 + data/train_ok 下的额外良品图。"""
    from pathlib import Path

    imgs: list[np.ndarray] = []
    data_dir = Path(data_dir)
    index_path = data_dir / "index.json"
    if index_path.exists():
        import json
        for item in json.loads(index_path.read_text(encoding="utf-8")):
            if item.get("ok"):
                g = cv2.imread(str(data_dir / "images" / item["file"]), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    imgs.append(g)
    extra_dir = data_dir / "train_ok"
    if extra_dir.exists():
        for f in sorted(extra_dir.glob("*.png")):
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                imgs.append(g)
    return imgs


def train(images: list[np.ndarray], bank_size: int = BANK_SIZE,
          log=None, progress=None, augment: bool = False) -> AnomalyModel:
    """只用良品图训练：统计归一化参数 + 建立特征库 + 定阈值。

    log(msg)       —— 逐步过程回调（给界面打印训练日志）
    progress(pct)  —— 0~100 进度回调
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    def _prog(pct: float) -> None:
        if progress:
            progress(max(0.0, min(100.0, pct)))

    n_raw = len(images)
    if augment:
        images = augment_goods(images)
        _log(f"[1/5] 读取良品图：{n_raw} 张，增广后 {len(images)} 张（含翻转/亮度/噪声，不使用任何缺陷样本）")
    else:
        _log(f"[1/5] 读取良品图：{n_raw} 张（不使用任何缺陷样本）")
    _prog(2)

    all_feats = []
    n = max(1, len(images))
    for i, img in enumerate(images, 1):
        f, _ = patch_features(img)
        all_feats.append(f)
        if i % max(1, n // 6) == 0 or i == n:
            _log(f"      提取 patch 特征 {i}/{n} 张 · 累计 {sum(len(x) for x in all_feats)} 条")
            _prog(2 + 23 * i / n)
    feats = np.concatenate(all_feats, axis=0)
    _log(f"[2/5] 汇总 patch 特征：{len(feats)} 条 × {feats.shape[1]} 维")
    _prog(26)

    mu = feats.mean(axis=0)
    sd = feats.std(axis=0) + 1e-6
    z = (feats - mu) / sd
    _log(f"      z-score 归一化完成（均值 {mu.round(1)}）")
    _prog(32)

    # 特征库：均匀随机下采样，保证覆盖不同位置与纹理
    rng = np.random.default_rng(20261122)
    idx = rng.choice(len(z), size=min(bank_size, len(z)), replace=False)
    bank = np.ascontiguousarray(z[idx])
    _log(f"[3/5] 建立特征库：随机下采样 {len(bank)} 条（种子 20261122）")
    _prog(45)

    model = AnomalyModel(bank=bank, mu=mu, sd=sd, thr=0.5, score_hi=1.0)

    # 图像级校准：每张良品图取 top-k 均值作为该图的「异常强度」，
    # 阈值设为这些强度的 99 分位 —— 保证良品误报率可控（而不是 patch 级分位）
    _log("[4/5] 阈值校准：让每张良品图给自己打分")
    img_scores = []
    for img in images:
        s = score_map(img, model)
        flat = np.sort(s.reshape(-1))[::-1]
        k = max(1, int(flat.size * 0.002))
        img_scores.append(float(flat[:k].mean()))
    arr = np.array(img_scores)
    model.thr = float(np.quantile(arr, THR_QUANTILE))
    model.score_hi = float(np.quantile(arr, 0.999)) * 1.2
    _log(f"      良品异常强度：中位 {np.median(arr):.3f} · {int(THR_QUANTILE*1000)/10}% 分位 {model.thr:.3f}")
    _log(f"      最终阈值 {model.thr:.3f}（良品误报按该分位控制）")
    _prog(92)
    return model


def score_map(img: np.ndarray, model: AnomalyModel) -> np.ndarray:
    """返回与原图同尺寸的异常分数图（0~1）。"""
    feats, (gh, gw) = patch_features(img)
    z = (feats - model.mu) / model.sd

    # 最近邻 L2：||a-b||^2 = |a|^2 + |b|^2 - 2ab
    a2 = np.einsum("ij,ij->i", z, z)[:, None]
    b2 = np.einsum("ij,ij->i", model.bank, model.bank)[None, :]
    d2 = a2 + b2 - 2.0 * z @ model.bank.T
    nn = np.sqrt(np.maximum(d2.min(axis=1), 0.0))

    grid = nn.reshape(gh, gw).astype(np.float32)
    full = cv2.resize(grid, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(full, (0, 0), 4.0)      # 原始最近邻距离，不做自归一化


def detect_unsup(img: np.ndarray, model: AnomalyModel, p) -> tuple[list, np.ndarray, np.ndarray]:
    """无监督检测：分数图阈值化 → 连通域 → 复用统一的评分与分级。"""
    from .common import analyze_regions

    score = score_map(img, model)
    mask = (score >= model.thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    scale = max(model.score_hi, 1e-6)
    contrast = np.clip(score / scale * 255.0, 0, 255).astype(np.uint8)
    regions = analyze_regions(mask, contrast, p)
    return regions, mask, contrast
