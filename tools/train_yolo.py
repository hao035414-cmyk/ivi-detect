"""YOLO26n 对照实验：用同一批标注训练一个深度检测器，与规则链路做对照。

用法：python -m tools.train_yolo [epochs] [imgsz]
产物：out/yolo_runs/yolo26n/ 下的权重与曲线；out/yolo_report.json 指标汇总
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "yolo" / "dataset.yaml"
WEIGHTS = ROOT / "weights" / "yolo26n.pt"
OUT = ROOT / "out"


def main() -> None:
    from ultralytics import YOLO

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 480

    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        workers=0,
        device="cpu",
        project=str(OUT / "yolo_runs"),
        name="yolo26n",
        exist_ok=True,
        amp=False,
        plots=False,
        verbose=True,
        seed=0,
    )

    best = OUT / "yolo_runs" / "yolo26n" / "weights" / "best.pt"
    m = YOLO(str(best if best.exists() else WEIGHTS))
    res = m.val(data=str(DATA_YAML), imgsz=imgsz, batch=8, device="cpu",
                workers=0, plots=False, verbose=False)

    report = {
        "model": "yolo26n",
        "epochs": epochs,
        "imgsz": imgsz,
        "params_m": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
        "mAP50": round(float(res.box.map50), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "per_class_map50": [round(float(x), 4) for x in res.box.maps[:4]] if res.box.maps is not None else None,
        "val_images": 62,
        "train_images": 62,
        "train_boxes": 158,
        "val_boxes": 152,
    }
    (OUT / "yolo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("RESULT " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
