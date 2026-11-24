"""把评测集导出成 YOLO 格式，用于深度模型对照实验。

产物：data/yolo/
  images/train/*.png  labels/train/*.txt
  images/val/*.png    labels/val/*.txt
  dataset.yaml

划分：按 index 序号奇偶分（偶数训练、奇数验证），与分类模块的划分口径一致，
      避免「训练集见过验证集」的口径问题。
类别：0=scratch 1=pit 2=stain 3=oil（与 core/classify.py 一致）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from core.imageio import imread, imwrite

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "yolo"
CLASSES = ["scratch", "pit", "stain", "oil"]


def main() -> None:
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_tr = n_va = boxes_tr = boxes_va = 0
    for k, item in enumerate(index):
        if item["ok"] or not item["defects"]:
            continue
        split = "train" if k % 2 == 0 else "val"
        src = DATA / "images" / item["file"]
        img = imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]

        dst_img = OUT / "images" / split / item["file"]
        if not dst_img.exists():
            imwrite(dst_img, img)

        lines = []
        for d in item["defects"]:
            bb = d.get("bbox")
            if not bb:
                continue
            x0, y0, x1, y1 = bb
            cx = (x0 + x1) / 2 / w
            cy = (y0 + y1) / 2 / h
            bw = max(1e-6, (x1 - x0) / w)
            bh = max(1e-6, (y1 - y0) / h)
            cid = CLASSES.index(d["type"]) if d["type"] in CLASSES else 0
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (OUT / "labels" / split / (src.stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        if split == "train":
            n_tr += 1
            boxes_tr += len(lines)
        else:
            n_va += 1
            boxes_va += len(lines)

    yaml = f"""path: {OUT.as_posix()}
train: images/train
val: images/val

names:
  0: scratch
  1: pit
  2: stain
  3: oil
"""
    (OUT / "dataset.yaml").write_text(yaml, encoding="utf-8")
    print(f"训练 {n_tr} 张 / {boxes_tr} 框　验证 {n_va} 张 / {boxes_va} 框")
    print(f"yaml -> {OUT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
