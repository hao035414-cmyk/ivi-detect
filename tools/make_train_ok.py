"""生成额外良品图，用于无监督训练的「训练集」（不进入评测集）。

为什么要单独建训练集：
  评测集 160 张里有 36 张良品，做训练太少；再合成 70 张良品单独存放，
  训练用 106 张，评测仍用原 160 张 —— 训练集与评测集分离，指标口径不变。

用法：python -m tools.make_train_ok [数量]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from core import dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "train_ok"
START_SEED = 5000


def main(n: int = 70) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(n):
        img, _, _ = dataset.make_one(START_SEED + i, 0)   # 0 处缺陷 = 良品
        cv2.imwrite(str(OUT / f"ok_{i:03d}.png"), img)
        made += 1
    total = len(list(OUT.glob("*.png")))
    print(f"生成良品图 {made} 张 → {OUT}（目录内共 {total} 张）")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 70)
