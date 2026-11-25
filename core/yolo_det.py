"""YOLO26n 推理封装（对照档）。

定位：只在界面里作为「第四档」演示用，不属于交付包依赖。
  权重 5.4 MB，依赖 ultralytics + torch（约 2.5 GB），换机器要重装。
  权重缺失或依赖未安装时，调用方应隐藏该档位，不影响主链路。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import Region, grade

CLASS_NAMES = ["scratch", "pit", "stain", "oil"]


class YoloDetector:
    def __init__(self, weights: Path, imgsz: int = 480, conf: float = 0.25):
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.conf = conf
        self._model = None

    @staticmethod
    def available(weights: Path) -> bool:
        """权重存在且 ultralytics 可导入，才认为可用。"""
        if not Path(weights).exists():
            return False
        try:
            import ultralytics  # noqa: F401
        except Exception:
            return False
        return True

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights))
            # 预热：第一次推理包含图优化与内存分配，先跑一次空图，
            # 避免界面上第一帧显示 1.8 s 的假耗时
            warm = np.zeros((480, 640, 3), np.uint8)
            self._model.predict(warm, imgsz=self.imgsz, device="cpu", verbose=False)
        return self._model

    def detect(self, img: np.ndarray, p) -> tuple[list[Region], np.ndarray, np.ndarray]:
        """返回 (regions, mask, contrast)，与其余链路接口一致。"""
        model = self._load()
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        conf_thr = max(0.15, min(float(p.score_thr), 0.9))   # 跟着界面滑块走，但不低于 0.15
        res = model.predict(bgr, imgsz=self.imgsz, conf=conf_thr,
                            device="cpu", verbose=False)[0]

        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        contrast = np.zeros((h, w), np.uint8)
        regions: list[Region] = []

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return regions, mask, contrast

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for i, ((x0, y0, x1, y1), cf, cid) in enumerate(zip(xyxy, confs, clss), 1):
            x0, y0 = int(max(0, x0)), int(max(0, y0))
            x1, y1 = int(min(w - 1, x1)), int(min(h - 1, y1))
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            area_px = bw * bh

            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            cv2.rectangle(contrast, (x0, y0), (x1, y1), int(255 * float(cf)), -1)

            regions.append(Region(
                rid=f"Y{i:02d}",
                x=x0, y=y0, w=bw, h=bh,
                cx=x0 + bw / 2, cy=y0 + bh / 2,
                area_px=area_px,
                contrast=55.0 * float(cf),
                solidity=1.0,
                score=float(cf),
                level=grade(area_px, 55.0 * float(cf), p.px_per_mm),
                px_per_mm=p.px_per_mm,
                kind=CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "",
            ))
        return regions, mask, contrast
