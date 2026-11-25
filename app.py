"""IVI-Detect · 工业表面缺陷智能检测平台（桌面端）

三个模块：
  1) 缺陷检测  —— 改进方案 / 基线 Otsu 一键切换，参数实时可调，结果可导出
  2) 字符识别  —— 喷码/批号 OCR（轻量 ONNX 引擎，缺依赖时回退模板匹配）
  3) 尺寸测量  —— 亚像素边缘 + 稳健拟合，输出尺寸、重复性与倾角

启动：  .venv\\Scripts\\python.exe app.py
截图：  .venv\\Scripts\\python.exe app.py --shot out/app.png --image data/images/0007_NG.png
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import (anomaly, baseline, classify, improved, measure as measure_mod,
                  metrics, ocr, yolo_det)  # noqa: E402
from core.common import Params, overlay  # noqa: E402
from core.imageio import imread, imwrite  # noqa: E402
from core.textdraw import get_font  # noqa: E402

# ---------------------------------------------------------------- 主题
C = {
    "bg": "#0E1116",
    "panel": "#161B23",
    "panel2": "#1B212B",
    "line": "#252C38",
    "text": "#E7EBF3",
    "muted": "#8A93A5",
    "dim": "#5E6779",
    "accent": "#3B82F6",
    "accent2": "#1D4ED8",
    "ok": "#2FBF71",
    "ng": "#E5484D",
    "warn": "#F2A83B",
    "cyan": "#22D3EE",
}
FONT = "Microsoft YaHei UI"


def f(size=10, bold=False):
    return (FONT, size, "bold" if bold else "normal")


class Card(tk.Frame):
    """带标题的卡片容器。"""

    def __init__(self, master, title: str, pad=12, **kw):
        super().__init__(master, bg=C["panel"], highlightbackground=C["line"],
                         highlightthickness=1, **kw)
        head = tk.Frame(self, bg=C["panel"])
        head.pack(fill="x", padx=pad, pady=(pad, 6))
        tk.Label(head, text=title, bg=C["panel"], fg=C["muted"],
                 font=f(9, True)).pack(side="left")
        self.body = tk.Frame(self, bg=C["panel"])
        self.body.pack(fill="both", expand=True, padx=pad, pady=(0, pad))


class Seg(tk.Frame):
    """分段选择按钮组。"""

    def __init__(self, master, items: list[tuple[str, str]], on_change, value=None,
                 height=30):
        super().__init__(master, bg=C["panel2"], highlightbackground=C["line"],
                         highlightthickness=1)
        self.on_change = on_change
        self.value = value or items[0][0]
        self.buttons: dict[str, tk.Label] = {}
        for i, (key, label) in enumerate(items):
            lb = tk.Label(self, text=label, bg=C["panel2"], fg=C["muted"], font=f(9),
                          padx=8, pady=4, cursor="hand2")
            lb.pack(side="left", fill="both", expand=True)
            lb.bind("<Button-1>", lambda e, k=key: self.set(k))
            self.buttons[key] = lb
        self._paint()

    def set(self, key, notify=True):
        if key not in self.buttons:
            return
        self.value = key
        self._paint()
        if notify:
            self.on_change(key)

    def _paint(self):
        for k, lb in self.buttons.items():
            on = k == self.value
            lb.configure(bg=C["accent"] if on else C["panel2"],
                         fg="#FFFFFF" if on else C["muted"],
                         font=f(9, on))


class Slider(tk.Frame):
    def __init__(self, master, label, lo, hi, init, step, fmt="{:.2f}",
                 on_change=None, width=190):
        super().__init__(master, bg=C["panel"])
        self.fmt, self.on_change, self.var_val = fmt, on_change, init
        top = tk.Frame(self, bg=C["panel"])
        top.pack(fill="x")
        tk.Label(top, text=label, bg=C["panel"], fg=C["text"], font=f(9)).pack(side="left")
        self.val = tk.Label(top, text=fmt.format(init), bg=C["panel"], fg=C["cyan"], font=f(9))
        self.val.pack(side="right")
        self.scale = tk.Scale(self, from_=lo, to=hi, resolution=step, orient="horizontal",
                              bg=C["panel"], fg=C["text"], troughcolor="#2A3241",
                              activebackground=C["accent"], highlightthickness=0,
                              bd=1, relief="flat", showvalue=False, sliderlength=16,
                              width=9, sliderrelief="flat", command=self._changed)
        self.scale.set(init)
        self.scale.pack(fill="x", pady=(0, 2))

    def _changed(self, v):
        v = float(v)
        changed = abs(v - self.var_val) > 1e-9
        self.var_val = v
        self.val.configure(text=self.fmt.format(v))
        if changed and self.on_change:
            self.on_change(v)

    def get(self):
        return self.var_val


def _enable_dpi():
    """开启 DPI 感知：高分屏下字体与截图都按物理像素渲染，投影时更清晰。"""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self, shot: str | None = None, start_image: str | None = None,
                 start_mode: str = "detect", start_algo: str = "improved"):
        super().__init__()
        self.title("IVI-Detect · 工业表面缺陷智能检测平台")
        self.configure(bg=C["bg"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1780, sw - 30), min(1100, sh - 80)
        self.geometry(f"{w}x{h}+{max(0,(sw-w)//2)}+26")
        self.minsize(1180, 760)

        self.params = Params()
        self.mode = start_mode
        self.algo = start_algo
        self._training = False
        self.yolo_weights = ROOT / "out" / "yolo_runs" / "yolo26n" / "weights" / "best.pt"
        self.yolo = None
        if yolo_det.YoloDetector.available(self.yolo_weights):
            self.yolo = yolo_det.YoloDetector(self.yolo_weights)
            # 后台预热：把 3.7 秒的加载与图优化挪到启动阶段，切档时不再卡
            threading.Thread(target=lambda: self._warm_yolo(), daemon=True).start()
        self.unsup_model = None
        try:
            self.unsup_model = anomaly.AnomalyModel.load(str(ROOT / "out" / "anomaly_model.npz"))
        except Exception as exc:  # 模型缺失时该模式不可用，不影响主链路
            self.unsup_model = None
            try:
                (ROOT / "out" / "unsup_load_error.txt").write_text(repr(exc), encoding="utf-8")
            except Exception:
                pass
        self.roi: tuple[int, int, int, int] | None = None
        self._drag: tuple[int, int] | None = None
        self._images = self._collect_images()
        self.idx = 0
        if start_image:
            p = Path(start_image)
            if p.exists():
                self._images.insert(0, p)
                self.idx = 0
        self.img: np.ndarray | None = None
        self.result: dict = {}
        self._photo = {}
        self._last_vis: dict[int, np.ndarray] = {}
        self._resize_job = None
        self._batch_running = False

        self._style()
        self._build()
        self._set_columns(self.mode)
        self._apply_mode_ui(self.mode)
        self.update()
        self.after(80, lambda: self._load(self.idx))
        self.after(400, self._redraw)
        if shot:
            self.after(1200, lambda: self._grab(shot))

    # ------------------------------------------------------------ 资源
    @staticmethod
    def _collect_images() -> list[Path]:
        files: list[Path] = []
        d = ROOT / "data" / "images"
        if d.exists():
            files += sorted(d.glob("*.png"))[:80]
        demo = ROOT / "out" / "demo"
        if demo.exists():
            files += sorted(demo.glob("*.png"))
        return files

    def _style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("Treeview", background=C["panel2"], fieldbackground=C["panel2"],
                     foreground=C["text"], borderwidth=0, rowheight=25, font=f(9))
        st.configure("Treeview.Heading", background=C["panel"], foreground=C["muted"],
                     borderwidth=0, font=f(8, True), padding=(4, 6))
        st.map("Treeview", background=[("selected", C["accent"])],
               foreground=[("selected", "#FFFFFF")])
        st.configure("Vertical.TScrollbar", background=C["panel2"], troughcolor=C["panel"],
                     borderwidth=0, arrowcolor=C["muted"])
        st.configure("TProgressbar", background=C["accent"], troughcolor=C["panel2"],
                     borderwidth=0, thickness=6)

    # ------------------------------------------------------------ 布局
    def _build(self):
        self._header()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=14, pady=(10, 8))
        body.columnconfigure(0, weight=0, minsize=252)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=356)
        body.rowconfigure(0, weight=1)

        self._left(body)
        self._center(body)
        self._right(body)
        self._footer()

    def _header(self):
        bar = tk.Frame(self, bg=C["panel"])
        bar.pack(fill="x")
        mark = tk.Label(bar, text="◈", bg=C["panel"], fg=C["accent"], font=(FONT, 21, "bold"))
        mark.pack(side="left", padx=(18, 10))
        box = tk.Frame(bar, bg=C["panel"])
        box.pack(side="left", pady=11)
        tk.Label(box, text="IVI-Detect", bg=C["panel"], fg=C["text"],
                 font=f(14, True)).pack(anchor="w")
        tk.Label(box, text="工业表面缺陷智能检测平台 · 缺陷检测 / 字符识别 / 尺寸测量",
                 bg=C["panel"], fg=C["muted"], font=f(8)).pack(anchor="w")
        tk.Frame(self, bg=C["accent"], height=2).pack(fill="x")

        right = tk.Frame(bar, bg=C["panel"])
        right.pack(side="right", padx=18)
        self.lbl_clock = tk.Label(right, text="", bg=C["panel"], fg=C["dim"], font=f(9))
        self.lbl_clock.pack(side="right", padx=(12, 0))
        badge = tk.Frame(right, bg=C["panel2"], highlightbackground=C["line"], highlightthickness=1)
        badge.pack(side="right")
        tk.Label(badge, text="● 引擎就绪", bg=C["panel2"], fg=C["ok"], font=f(8)).pack(padx=10, pady=4)

    def _left(self, body):
        wrap = tk.Frame(body, bg=C["bg"])
        wrap.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        wrap.columnconfigure(0, weight=1)
        c1 = Card(wrap, "作业模块")
        c1.grid(row=0, column=0, sticky="ew")
        self.seg_mode = Seg(c1.body, [("detect", "缺陷检测"), ("ocr", "字符识别"),
                                      ("measure", "尺寸测量")], self._on_mode, self.mode)
        self.seg_mode.pack(fill="x")

        c2 = Card(wrap, "算法方案")
        self.card_algo = c2
        c2.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        opts = [("improved", "规则链路"), ("unsup", "无监督"), ("yolo", "YOLO"),
                ("baseline", "基线 Otsu")]
        if self.unsup_model is None:
            opts = [o for o in opts if o[0] != "unsup"]
        if self.yolo is None:
            opts = [o for o in opts if o[0] != "yolo"]
        self.seg_algo = Seg(c2.body, opts, self._on_algo, self.algo)
        self.seg_algo.pack(fill="x")
        self.lbl_algo_note = tk.Label(c2.body, text="光照归一化 + 双频通道 + 代价阈值",
                                      bg=C["panel"], fg=C["dim"], font=f(8),
                                      wraplength=210, justify="left")
        self.lbl_algo_note.pack(anchor="w", pady=(6, 0))

        c3 = Card(wrap, "检测参数")
        self.card_param = c3
        c3.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        p = self.params
        self.sl_area = Slider(c3.body, "最小缺陷面积 (px)", 20, 400, p.min_area, 5, "{:.0f}",
                              lambda v: self._on_param("min_area", int(v)))
        self.sl_area.pack(fill="x", pady=(0, 6))
        self.sl_score = Slider(c3.body, "判定阈值 score_thr", 0.20, 0.90, p.score_thr, 0.01, "{:.2f}",
                               lambda v: self._on_param("score_thr", float(v)))
        self.sl_score.pack(fill="x", pady=(0, 6))
        self.sl_kh = Slider(c3.body, "高频灵敏度 k_h", 1.0, 4.5, p.k_h, 0.1, "{:.1f}",
                            lambda v: self._on_param("k_h", float(v)))
        self.sl_kh.pack(fill="x", pady=(0, 6))
        self.sl_kl = Slider(c3.body, "低频灵敏度 k_l", 1.5, 5.5, p.k_l, 0.1, "{:.1f}",
                            lambda v: self._on_param("k_l", float(v)))
        self.sl_kl.pack(fill="x", pady=(0, 6))
        self.sl_cal = Slider(c3.body, "标定 (px / mm)", 5, 30, p.px_per_mm, 0.5, "{:.1f}",
                             lambda v: self._on_param("px_per_mm", float(v)))
        self.sl_cal.pack(fill="x")

        c4 = Card(wrap, "操作")
        c4.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self._button(c4.body, "打开图像", self._open, accent=True).pack(fill="x", pady=(0, 6))
        row = tk.Frame(c4.body, bg=C["panel"])
        row.pack(fill="x", pady=(0, 6))
        self._button(row, "◀ 上一张", self._prev, small=True).pack(side="left", fill="x", expand=True)
        self._button(row, "下一张 ▶", self._next, small=True).pack(side="left", fill="x",
                                                                   expand=True, padx=(6, 0))
        self._button(c4.body, "批量评测全库", self._batch, accent=True).pack(fill="x", pady=(0, 6))
        self._button(c4.body, "训练无监督模型", self._train_unsup).pack(fill="x", pady=(0, 6))
        self._button(c4.body, "导出当前结果", self._export).pack(fill="x")

    @staticmethod
    def _button(master, text, cmd, accent=False, small=False):
        b = tk.Label(master, text=text, bg=C["accent"] if accent else C["panel2"],
                     fg="#FFFFFF" if accent else C["text"],
                     font=f(9 if small else 10, False), pady=5 if small else 7,
                     cursor="hand2", highlightbackground=C["line"], highlightthickness=0)
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.configure(bg=C["accent2"] if accent else C["line"]))
        b.bind("<Leave>", lambda e: b.configure(bg=C["accent"] if accent else C["panel2"]))
        return b

    def _center(self, body):
        wrap = tk.Frame(body, bg=C["bg"])
        wrap.grid(row=0, column=1, sticky="nsew")
        wrap.rowconfigure(1, weight=1)
        wrap.columnconfigure(0, weight=1)

        top = tk.Frame(wrap, bg=C["bg"])
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.lbl_file = tk.Label(top, text="-", bg=C["bg"], fg=C["text"], font=f(10, True))
        self.lbl_file.pack(side="left")
        self.lbl_meta = tk.Label(top, text="", bg=C["bg"], fg=C["dim"], font=f(9))
        self.lbl_meta.pack(side="left", padx=(10, 0))
        self.lbl_tip = tk.Label(top, text="← → 切换图像 · 拖动框选 ROI · 参数实时生效",
                                bg=C["bg"], fg=C["dim"], font=f(8))
        self.lbl_tip.pack(side="right")

        self.canv_titles = ["原始图像", "检测结果", "反差 / 掩膜"]
        grid = tk.Frame(wrap, bg=C["bg"])
        grid.grid(row=1, column=0, sticky="nsew")
        grid.columnconfigure(0, weight=1, uniform="c")
        grid.columnconfigure(1, weight=1, uniform="c")
        grid.rowconfigure(0, weight=1)
        self.canvases: list[tk.Canvas] = []
        for i, title in enumerate(self.canv_titles[:2]):
            cell = tk.Frame(grid, bg=C["panel"], highlightbackground=C["line"],
                            highlightthickness=1)
            cell.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            head = tk.Frame(cell, bg=C["panel"])
            head.pack(fill="x", padx=8, pady=(6, 2))
            tk.Label(head, text=title, bg=C["panel"], fg=C["muted"],
                     font=f(8, True)).pack(side="left")
            cv = tk.Canvas(cell, bg="#0B0E13", highlightthickness=0)
            cv.pack(fill="both", expand=True, padx=1, pady=(0, 1))
            self._bind_canvas(cv)
            self.canvases.append(cv)

        strip = tk.Frame(grid, bg=C["panel"], highlightbackground=C["line"],
                         highlightthickness=1)
        strip.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        head = tk.Frame(strip, bg=C["panel"])
        head.pack(fill="x", padx=8, pady=(5, 1))
        tk.Label(head, text=self.canv_titles[2], bg=C["panel"], fg=C["muted"],
                 font=f(8, True)).pack(side="left")
        self.lbl_strip = tk.Label(head, text="inferno 伪彩 = 反差强度 / 灰度 = 检出掩膜",
                                  bg=C["panel"], fg=C["dim"], font=f(8))
        self.lbl_strip.pack(side="right")
        cv3 = tk.Canvas(strip, bg="#0B0E13", height=118, highlightthickness=0)
        cv3.pack(fill="x", padx=1, pady=(0, 1))
        self._bind_canvas(cv3)
        self.canvases.append(cv3)

        self.progress = ttk.Progressbar(wrap, mode="determinate", maximum=100)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.progress.grid_remove()

    def _bind_canvas(self, cv):
        cv.bind("<Configure>", lambda e: self._on_resize())
        cv.bind("<Button-1>", self._roi_start)
        cv.bind("<B1-Motion>", self._roi_drag)
        cv.bind("<ButtonRelease-1>", self._roi_end)

    def _right(self, body):
        wrap = tk.Frame(body, bg=C["bg"])
        wrap.grid(row=0, column=2, sticky="nse", padx=(10, 0))

        card = tk.Frame(wrap, bg=C["panel"], highlightbackground=C["line"], highlightthickness=1)
        card.pack(fill="x")
        inner = tk.Frame(card, bg=C["panel"])
        inner.pack(fill="x", padx=14, pady=12)
        self.lbl_verdict = tk.Label(inner, text="—", bg=C["panel"], fg=C["muted"],
                                    font=(FONT, 30, "bold"))
        self.lbl_verdict.pack(anchor="w")
        self.lbl_verdict_sub = tk.Label(inner, text="等待检测", bg=C["panel"], fg=C["dim"],
                                        font=f(9))
        self.lbl_verdict_sub.pack(anchor="w", pady=(2, 0))

        stats = tk.Frame(card, bg=C["panel"])
        stats.pack(fill="x", padx=14, pady=(0, 12))
        self.stat_labels = {}
        self.stat_names = {}
        for i, (key, name) in enumerate([("count", "检出"), ("level", "最高等级"),
                                         ("ms", "耗时"), ("fps", "吞吐")]):
            cell = tk.Frame(stats, bg=C["panel2"])
            cell.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 4, 0))
            stats.columnconfigure(i, weight=1)
            tk.Label(cell, text=name, bg=C["panel2"], fg=C["dim"], font=f(7)).pack(
                pady=(6, 0))
            lb = tk.Label(cell, text="—", bg=C["panel2"], fg=C["text"], font=f(11, True))
            lb.pack(pady=(0, 6))
            self.stat_labels[key] = lb
            self.stat_names[key] = cell.winfo_children()[0]

        tcard = Card(wrap, "缺陷明细")
        tcard.pack(fill="both", expand=True, pady=(10, 0))
        self.tcard_title = tcard
        cols = ("c1", "c2", "c3", "c4")
        self.tree = ttk.Treeview(tcard.body, columns=cols, show="headings", height=9)
        self._set_columns("detect")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("l3", foreground=C["ng"])
        self.tree.tag_configure("l2", foreground=C["warn"])
        self.tree.tag_configure("l1", foreground=C["ok"])
        self.tree.tag_configure("row", foreground=C["text"])

        self.txt_info = tk.Text(wrap, height=8, width=1, bg=C["panel"], fg=C["muted"], font=f(9),
                                bd=0, highlightthickness=1, highlightbackground=C["line"],
                                padx=10, pady=8, wrap="word")
        self.txt_info.pack(fill="x", pady=(10, 0))
        self.txt_info.insert("1.0", "就绪。选择模块后打开图像，或直接使用内置样例。")
        self.txt_info.configure(state="disabled")

    def _set_columns(self, mode):
        spec = {
            "detect": [("编号", 48), ("类型", 58), ("等级", 58), ("面积 mm²", 88)],
            "ocr": [("段", 40), ("识别文本", 130), ("字数", 52), ("置信度", 66)],
            "measure": [("项目", 70), ("数值", 110), ("像素/备注", 82), ("标记", 60)],
        }[mode]
        for (title, w), col in zip(spec, ("c1", "c2", "c3", "c4")):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor="center")
        names = {
            "detect": ("检出", "最高等级", "耗时", "吞吐"),
            "ocr": ("字符数", "置信度", "耗时", "吞吐"),
            "measure": ("宽度mm", "高度mm", "耗时", "重复性"),
        }[mode]
        for (k, _), nm in zip(self.stat_labels.items(), names):
            self.stat_names[k].configure(text=nm)

    def _footer(self):
        bar = tk.Frame(self, bg=C["panel"], height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.lbl_status = tk.Label(bar, text="就绪", bg=C["panel"], fg=C["muted"], font=f(8))
        self.lbl_status.pack(side="left", padx=16)
        tk.Label(bar, text="IVI-Detect v1.0 · Python + OpenCV · 本地离线运行",
                 bg=C["panel"], fg=C["dim"], font=f(8)).pack(side="right", padx=16)

        self.bind("<Left>", lambda e: self._prev())
        self.bind("<Right>", lambda e: self._next())
        self.bind("<r>", lambda e: self._run())

    # ------------------------------------------------------------ 交互
    def _on_mode(self, mode):
        self.mode = mode
        self.roi = None
        self._set_columns(mode)
        self._apply_mode_ui(mode)
        self._load(self.idx)

    def _apply_mode_ui(self, mode):
        """按模块收起无效控件，避免界面上出现用不到的参数。"""
        if mode == "detect":
            self.card_algo.grid()
            self.card_param.grid()
        else:
            self.card_algo.grid_remove()
        if mode == "ocr":
            self.card_param.grid_remove()
        else:
            self.card_param.grid()
        self.lbl_tip.configure(
            text="← → 切换图像 · 拖动框选 ROI · 参数实时生效" if mode == "detect"
            else "← → 切换图像 · 在图像上拖动框选 ROI")

    def _on_algo(self, algo):
        self.algo = algo
        self.lbl_algo_note.configure(
            text="光照归一化 + 双频通道 + 代价阈值" if algo == "improved"
            else "良品特征库 kNN 距离 · 无需标注" if algo == "unsup"
            else "YOLO26n 对照档 · mAP50 0.952 · 需 2.5 GB 依赖" if algo == "yolo"
            else "全局 Otsu 阈值 + 形态学（对照基线）")
        self._run()

    def _on_param(self, key, val):
        setattr(self.params, key, val)
        self._run()

    def _status(self, text):
        self.lbl_status.configure(text=text)

    def _info(self, text):
        self.txt_info.configure(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", text)
        self.txt_info.configure(state="disabled")

    # ------------------------------------------------------------ 加载与运行
    def _load(self, i):
        if not self._images:
            self._info("未找到图像。请把图片放入 data/images/ 或点击「打开图像」。")
            return
        self.idx = i % len(self._images)
        path = self._images[self.idx]
        img = imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._info(f"无法读取：{path.name}")
            return
        self.img = img
        self.lbl_file.configure(text=path.name)
        h, w = img.shape[:2]
        self.lbl_meta.configure(text=f"{w}×{h} · 灰度 · {path.parent.name}/")
        self._run()

    def _prev(self):
        self._load(self.idx - 1)

    def _next(self):
        self._load(self.idx + 1)

    def _open(self):
        p = filedialog.askopenfilename(
            title="选择图像", initialdir=str(ROOT / "data" / "images"),
            filetypes=[("图像", "*.png *.jpg *.jpeg *.bmp"), ("全部", "*.*")])
        if p:
            self._images.insert(0, Path(p))
            self._load(0)

    def _run(self):
        if self.img is None:
            return
        # YOLO 档：先把模型加载完再开始计时，避免把 3.7 s 的加载算进单帧耗时
        if self.mode == "detect" and self.algo == "yolo" and self.yolo is not None:
            self.yolo._load()
        t0 = time.perf_counter()
        if self.mode == "detect":
            self._run_detect()
        elif self.mode == "ocr":
            self._run_ocr()
        else:
            self._run_measure()
        ms = (time.perf_counter() - t0) * 1000
        self.stat_labels["ms"].configure(text=f"{ms:.0f}ms")
        if self.mode != "measure":
            self.stat_labels["fps"].configure(text=f"{1000 / max(ms, 1e-6):.0f}")
        self._status(f"{self.mode} · {self.algo if self.mode == 'detect' else '默认'} · "
                     f"{ms:.0f} ms · {self._images[self.idx].name}")

    def _run_detect(self):
        img = self.img
        p = self.params
        scaled = False
        if self.algo == "unsup" and self.unsup_model is not None:
            regs, mask, contrast = anomaly.detect_unsup(img, self.unsup_model, p)
        elif self.algo == "yolo" and self.yolo is not None:
            regs, mask, contrast = self.yolo.detect(img, p)
        elif self.algo == "baseline":
            regs, mask, contrast = baseline.detect(img, Params(**{**p.__dict__}))
        else:
            # 标定偏离基准尺度 20% 以上时自动走尺度归一化链路
            scaled = abs(p.px_per_mm / improved.CANONICAL_PX_PER_MM - 1.0) > 0.2
            fn = improved.detect_adaptive if scaled else improved.detect
            regs, mask, contrast = fn(img, p)

        # 统一补缺陷类型：基线、无监督两条链路原本没挂分类
        for r in regs:
            if not r.kind:
                try:
                    feat = classify.region_features(r, contrast, mask, p.px_per_mm)
                    r.kind = classify.classify(feat)
                except Exception:
                    r.kind = ""

        vis = overlay(img, regs)
        contrast_vis = cv2.applyColorMap(
            cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_INFERNO)
        contrast_vis = cv2.addWeighted(contrast_vis, 0.78,
                                       cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.22, 0)

        self._show(0, img)
        self._show(1, vis)
        self._show(2, contrast_vis)

        n = len(regs)
        lv = max((r.level for r in regs), default=0)
        if n == 0:
            self.lbl_verdict.configure(text="OK", fg=C["ok"])
            self.lbl_verdict_sub.configure(text="未检出缺陷 · 放行")
        else:
            self.lbl_verdict.configure(text="NG", fg=C["ng"])
            self.lbl_verdict_sub.configure(
                text=f"检出 {n} 处缺陷 · 最高等级 {['-', '轻微', '中等', '严重'][lv]}")
        self.stat_labels["count"].configure(text=str(n))
        self.stat_labels["level"].configure(
            text=["—", "轻微", "中等", "严重"][lv] if lv else "—")

        self.tree.delete(*self.tree.get_children())
        for r in regs:
            self.tree.insert("", "end", tags=(f"l{r.level}",),
                             values=(r.rid, r.kind_cn, r.level_cn, f"{r.area_mm2:.2f}"))

        total = sum(r.area_mm2 for r in regs)
        algo_cn = {"improved": "规则链路", "unsup": "无监督（良品训练）",
                   "yolo": "YOLO26n（对照档）", "baseline": "基线 Otsu"}.get(self.algo, self.algo)
        scale_note = "｜已启用尺度归一化（按标定重采样到 12 px/mm）" if scaled else ""
        self._info(f"方案：{algo_cn}{scale_note}\n"
                   f"检出 {n} 处，总面积 {total:.2f} mm²，最大 {max([r.area_mm2 for r in regs], default=0):.2f} mm²\n"
                   f"参数：min_area={p.min_area}px / score_thr={p.score_thr:.2f} / "
                   f"k_h={p.k_h:.1f} / k_l={p.k_l:.1f} / {p.px_per_mm:.1f}px·mm⁻¹\n"
                   f"判定：{'放行' if n == 0 else '需处置（严重件直接判废）'}")
        self.result = {"type": "detect", "regions": [r.as_dict() for r in regs],
                       "algo": self.algo, "file": self._images[self.idx].name}

    def _run_ocr(self):
        img = self.img
        res = ocr.recognize(img, self.roi)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for box in res.get("boxes", []):
            pts = np.array(box, np.int32)
            cv2.polylines(vis, [pts], True, (60, 200, 255), 2)
        self._show(0, img)
        self._show(1, vis)
        self._show(2, res["binary"])

        ok = res["verdict"] == "PASS"
        self.lbl_verdict.configure(text="PASS" if ok else "REVIEW",
                                   fg=C["ok"] if ok else C["warn"])
        self.lbl_verdict_sub.configure(
            text=f"识别文本：{res['text'] or '（无字符）'} · 置信度 {res['conf']:.3f}")
        self.stat_labels["count"].configure(text=str(res["n_chars"]))
        self.stat_labels["level"].configure(text=f"{res['conf']:.2f}")

        self.tree.delete(*self.tree.get_children())
        for i, seg in enumerate(res["per_char"], 1):
            self.tree.insert("", "end", tags=("row",),
                             values=(i, seg["ch"], len(seg["ch"]), f"{seg['conf']:.2f}"))
        self._info(f"引擎：{'轻量 ONNX（PP-OCR）' if res['engine'] == 'onnx' else '模板匹配回退'}\n"
                   f"字符数 {res['n_chars']}，平均置信度 {res['conf']:.3f}\n"
                   f"判定：{'通过' if ok else '需人工复核（置信度不足或字符数异常）'}\n"
                   f"适用边界：水平排版、字符间距均匀；曲面/点阵/严重倾斜需改用更高阶模型。")
        self.result = {"type": "ocr", "text": res["text"], "conf": res["conf"],
                       "file": self._images[self.idx].name}

    def _run_measure(self):
        img = self.img
        res = measure_mod.measure(img, self.roi, self.params.px_per_mm)
        vis = measure_mod.draw_measure(img, res)
        from core.textdraw import put_text
        put_text(vis, f"{res['w_mm']:.2f} × {res['h_mm']:.2f} mm", (10, 10),
                 size=16, color=(255, 220, 90), bg=(18, 22, 28))
        edges = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 40, 110)
        self._show(0, img)
        self._show(1, vis)
        self._show(2, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

        ok = res["confident"]
        self.lbl_verdict.configure(text=f"{res['w_mm']:.1f}", fg=C["cyan"])
        self.lbl_verdict_sub.configure(
            text=f"宽度 mm（高度 {res['h_mm']:.2f} mm）· 倾角 {res['angle_deg']:+.2f}°")
        self.stat_labels["count"].configure(text=f"{res['w_mm']:.2f}")
        self.stat_labels["level"].configure(text=f"{res['h_mm']:.2f}")
        self.stat_labels["fps"].configure(text=f"{res['repeatability_um']:.0f}µm")

        self.tree.delete(*self.tree.get_children())
        rows = [("宽度", f"{res['w_mm']:.3f} mm", f"{res['w_px']:.1f} px", "—"),
                ("高度", f"{res['h_mm']:.3f} mm", f"{res['h_px']:.1f} px", "—"),
                ("倾角", f"{res['angle_deg']:+.3f}°", "—", "—"),
                ("重复性", f"{res['repeatability_um']:.1f} µm", "—",
                 "1σ" if res["confident"] else "低")]
        for r in rows:
            self.tree.insert("", "end", tags=("row",), values=r)
        self._info(f"标定：{res['px_per_mm']:.2f} px/mm；有效边缘占比 "
                   f"{res['valid_ratio'][0]:.0%} / {res['valid_ratio'][1]:.0%}\n"
                   f"边缘用梯度峰值 + 抛物线拟合定位（亚像素），逐行/逐列中位数 + MAD 剔除\n"
                   f"重复性 1σ = {res['repeatability_um']:.1f} µm（{res['repeatability_um']/1000/res['w_mm']*100:.3f}% FS）\n"
                   f"判定：{'可信' if res['confident'] else '边缘不足，请重新框选 ROI 或调整光照'}")
        self.result = {"type": "measure", **res}

    # ------------------------------------------------------------ 显示
    def _on_resize(self):
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(90, self._redraw)

    def _redraw(self):
        self._resize_job = None
        for slot, img in self._last_vis.items():
            cv_ = self.canvases[slot]
            if cv_.winfo_width() > 10:
                self._show(slot, img)

    def _show(self, slot: int, img_bgr: np.ndarray):
        self._last_vis[slot] = img_bgr
        cv_ = self.canvases[slot]
        cv_.update_idletasks()
        cw = max(80, cv_.winfo_width())
        ch = max(60, cv_.winfo_height())
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]
        if slot == 2:      # 底部条：按填充方式裁切，避免两侧大片留黑
            s = max(cw / w, ch / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            big = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            ox, oy = (nw - cw) // 2, (nh - ch) // 2
            img_bgr = big[max(0, oy):max(0, oy) + ch, max(0, ox):max(0, ox) + cw]
            h, w = img_bgr.shape[:2]
            nw, nh = w, h
        else:
            s = min(cw / w, ch / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
        pil = Image.fromarray(cv2.cvtColor(cv2.resize(img_bgr, (nw, nh),
                                                      interpolation=cv2.INTER_AREA),
                                           cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(pil)
        self._photo[slot] = photo
        cv_.delete("all")
        cv_.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        cv_._img_geom = (cw, ch, nw, nh, w, h)

    # ------------------------------------------------------------ ROI
    def _canvas_to_img(self, cv_, x, y):
        if not hasattr(cv_, "_img_geom"):
            return None
        cw, ch, nw, nh, w, h = cv_._img_geom
        ox, oy = (cw - nw) / 2, (ch - nh) / 2
        ix = int((x - ox) / max(nw, 1) * w)
        iy = int((y - oy) / max(nh, 1) * h)
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    def _roi_start(self, e):
        cv_ = e.widget
        p = self._canvas_to_img(cv_, e.x, e.y)
        if p:
            self._drag = p

    def _roi_drag(self, e):
        if not self._drag:
            return
        p = self._canvas_to_img(e.widget, e.x, e.y)
        if not p:
            return
        cv_.delete("roi")
        x0, y0 = self._drag
        cv_.create_rectangle(x0 * 0 + 0, 0, 0, 0, outline="", tags="roi")  # placeholder
        cw, ch, nw, nh, w, h = e.widget._img_geom
        ox, oy = (cw - nw) / 2, (ch - nh) / 2
        sx, sy = nw / w, nh / h
        e.widget.delete("roi")
        e.widget.create_rectangle(ox + x0 * sx, oy + y0 * sy, ox + p[0] * sx, oy + p[1] * sy,
                                  outline=C["accent"], width=2, dash=(4, 3), tags="roi")

    def _roi_end(self, e):
        if not self._drag:
            return
        p = self._canvas_to_img(e.widget, e.x, e.y)
        start = self._drag
        self._drag = None
        if not p:
            return
        x0, y0 = min(start[0], p[0]), min(start[1], p[1])
        x1, y1 = max(start[0], p[0]), max(start[1], p[1])
        if x1 - x0 < 12 or y1 - y0 < 12:
            self.roi = None
            e.widget.delete("roi")
            self._run()
            return
        self.roi = (x0, y0, x1, y1)
        self._run()

    # ------------------------------------------------------------ 批量评测
    def _warm_yolo(self):
        try:
            self.yolo._load()
        except Exception:
            pass

    def _train_unsup(self):
        """用库内良品图重新训练无监督模型：弹出过程窗口，逐步打印日志。"""
        if getattr(self, "_training", False):
            return
        self._training = True

        dlg = tk.Toplevel(self)
        dlg.title("训练无监督模型 · 过程")
        dlg.geometry("660x430")
        dlg.configure(bg=C["panel"])
        dlg.transient(self)

        tk.Label(dlg, text="只用良品图训练，不使用任何缺陷样本", bg=C["panel"],
                 fg=C["text"], font=f(11, bold=True)).pack(anchor="w", padx=18, pady=(14, 2))
        tk.Label(dlg, text="五步：载入良品图 → patch 特征 → 特征库 → 阈值校准 → 保存模型",
                 bg=C["panel"], fg=C["muted"], font=f(9)).pack(anchor="w", padx=18)

        pb = ttk.Progressbar(dlg, mode="determinate", maximum=100)
        pb.pack(fill="x", padx=18, pady=(10, 8))

        box = tk.Text(dlg, bg="#0F1724", fg="#D7E2F0", font=("Consolas", 9),
                      relief="flat", padx=10, pady=8, height=14)
        box.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        box.insert("end", "训练开始…\n")
        box.configure(state="disabled")

        foot = tk.Label(dlg, text="训练期间可继续看别的图，完成后自动加载新模型",
                        bg=C["panel"], fg=C["muted"], font=f(9))
        foot.pack(anchor="w", padx=18, pady=(0, 12))

        def push(msg: str) -> None:
            def _do():
                box.configure(state="normal")
                box.insert("end", msg + "\n")
                box.see("end")
                box.configure(state="disabled")
            self.after(0, _do)

        def set_pct(pct: float) -> None:
            self.after(0, lambda: pb.configure(value=pct))

        self.progress.grid()
        self.progress["value"] = 0
        self._status("训练中：只用良品图建立特征库…")

        def work():
            import time as _t
            t0 = _t.perf_counter()
            try:
                ok_imgs = anomaly.load_train_goods(ROOT / "data")
                model = anomaly.train(ok_imgs, log=push, progress=set_pct)
                push("[5/5] 保存模型 out/anomaly_model.npz")
                model.save(str(ROOT / "out" / "anomaly_model.npz"))
                ms = (_t.perf_counter() - t0) * 1000
                push(f"完成：{len(ok_imgs)} 张良品 · {ms:.0f} ms · 特征库 {model.bank.shape[0]} 条 · 阈值 {model.thr:.3f}")
                self.after(0, lambda: self._train_done(len(ok_imgs), ms, model, None))
            except Exception as exc:  # noqa: BLE001
                push(f"失败：{exc}")
                self.after(0, lambda: self._train_done(0, 0, None, exc))

        threading.Thread(target=work, daemon=True).start()

    def _train_done(self, n_ok, ms, model, exc):
        self._training = False
        self.progress.grid_remove()
        if exc is not None or model is None:
            self._status(f"训练失败：{exc}")
            return
        self.unsup_model = model
        self._status(f"训练完成：{n_ok} 张良品 · {ms:.0f} ms · 特征库 {model.bank.shape[0]} 条")
        self._info(f"无监督模型已更新（本次训练 {ms:.0f} ms）\n"
                   f"训练样本：{n_ok} 张良品，未使用任何缺陷图\n"
                   f"特征：16×16 patch · 5 维统计量 · 库容量 {model.bank.shape[0]}\n"
                   f"阈值：{model.thr:.3f}（由良品分数分位校准）\n"
                   f"模型文件：out/anomaly_model.npz")

    def _batch(self):
        if self._batch_running:
            return
        self._batch_running = True
        self.progress.grid()
        self.progress["value"] = 0
        self._status("批量评测进行中：基线 vs 改进，全库 160 张…")
        self.tree.delete(*self.tree.get_children())
        self._info("正在跑全库评测，请稍候（约 15 秒）。\n基线 Otsu 与改进方案使用同一批图像、"
                   "同一后处理与匹配准则。")

        def work():
            try:
                data = ROOT / "data"
                rows = []
                for name, fn, p in [("基线 Otsu", baseline.detect, Params(otsu_bias=0)),
                                    ("改进方案", improved.detect, self.params)]:
                    self.after(0, lambda n=name: self._status(f"评测中：{n} …"))
                    m = metrics.evaluate(fn, p, data, collect=True, limit=160)
                    rows.append((name, m))
                self.after(0, lambda: self._batch_done(rows))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._info(f"批量评测失败：{exc}"))
            finally:
                self.after(0, self._batch_end)

        threading.Thread(target=work, daemon=True).start()

    def _batch_end(self):
        self._batch_running = False
        self.progress.grid_remove()

    def _batch_done(self, rows):
        self.tree.delete(*self.tree.get_children())
        for name, m in rows:
            self.tree.insert("", "end", tags=("row",),
                             values=(name, f"F1 {m['f1']:.3f}",
                                     f"P {m['precision']:.1%}", f"R {m['recall']:.1%}"))
            self.tree.insert("", "end", tags=("row",),
                             values=("", f"漏检 {m['miss_rate']:.1%}",
                                     f"误报 {m['ok_false_alarm']:.1%}", f"{m['avg_ms']:.0f}ms"))
        best = max(rows, key=lambda kv: kv[1]["f1"])
        self.lbl_verdict.configure(text="EP", fg=C["accent"])
        self.lbl_verdict_sub.configure(text=f"批量评测完成 · 最优方案：{best[0]}")
        self.stat_labels["count"].configure(text=str(rows[-1][1]["images"]))
        self.stat_labels["level"].configure(text=f"{rows[-1][1]['f1']:.2f}")

        lines = []
        for name, m in rows:
            lines.append(f"{name}：P {m['precision']:.1%} / R {m['recall']:.1%} / "
                         f"F1 {m['f1']:.3f} / 漏检 {m['miss_rate']:.1%} / "
                         f"误报 {m['ok_false_alarm']:.1%} / {m['avg_ms']:.0f} ms·张⁻¹")
        d = rows[-1][1]
        pt = d.get("per_type", {})
        if pt:
            cn = {"scratch": "划痕", "pit": "凹坑", "stain": "污渍", "oil": "油污"}
            lines.append("逐类召回：" + " · ".join(
                f"{cn.get(k, k)} {v['tp']/max(1,v['gt']):.1%}" for k, v in pt.items()))
        self._info("\n".join(lines))
        self._status("批量评测完成")
        self.result = {"type": "batch", "rows": [(n, m) for n, m in rows]}

    # ------------------------------------------------------------ 导出
    def _export(self):
        if self.img is None or not self.result:
            messagebox.showinfo("导出", "还没有结果可导出。")
            return
        out = ROOT / "out" / "export"
        out.mkdir(parents=True, exist_ok=True)
        stem = self._images[self.idx].stem
        if self.mode == "detect":
            regs = self.result["regions"]
            png = out / f"{stem}_detect.png"
            vis = self._last_vis.get(1)
            if vis is not None:
                imwrite(str(png), vis)
            csv_path = out / f"{stem}_defects.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["编号", "类型", "等级", "面积mm2", "长度mm", "x", "y", "w", "h", "置信度"])
                for r in regs:
                    w.writerow([r["rid"], r["kind_cn"], r["level_cn"], round(r["area_mm2"], 3),
                                round(r["length_mm"], 2), r["x"], r["y"], r["w"], r["h"],
                                r["score"]])
            messagebox.showinfo("导出", f"已导出：\n{png}\n{csv_path}")
        else:
            txt = out / f"{stem}_{self.mode}.txt"
            txt.write_text(str(self.result), encoding="utf-8")
            messagebox.showinfo("导出", f"已导出：\n{txt}")

    # ------------------------------------------------------------ 截图
    def _grab(self, path: str):
        self.update()
        try:
            from PIL import ImageGrab
            full = ImageGrab.grab()
            s = full.width / max(1, self.winfo_screenwidth())   # 物理/逻辑像素比
            x, y = self.winfo_rootx(), self.winfo_rooty()
            w, h = self.winfo_width(), self.winfo_height()
            box = (int(x * s), int(y * s), int((x + w) * s), int((y + h) * s))
            img = full.crop(box)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
            self._status(f"已保存截图 {path}")
        except Exception as exc:  # noqa: BLE001
            self._status(f"截图失败：{exc}")
        self.after(400, self.destroy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--mode", default="detect", choices=["detect", "ocr", "measure"])
    ap.add_argument("--algo", default="improved", choices=["improved", "unsup", "yolo", "baseline"])
    args = ap.parse_args()
    _enable_dpi()
    app = App(shot=args.shot, start_image=args.image, start_mode=args.mode,
              start_algo=args.algo)
    app.mainloop()


if __name__ == "__main__":
    main()
