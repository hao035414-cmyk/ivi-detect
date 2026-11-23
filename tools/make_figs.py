"""生成 PPT 用示意图（算法流程、判据示意）。

产物 out/fig_flow.png / out/fig_judge.png
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)

INK = (23, 32, 48)
MUTED = (100, 116, 139)
BLUE = (59, 130, 246)
BLUE_D = (30, 64, 175)
BLUE_L = (232, 240, 254)
AMBER = (245, 158, 11)
AMBER_L = (254, 243, 199)
GREEN = (16, 145, 96)
GREEN_L = (220, 246, 236)
GRAY_L = (244, 246, 250)
LINE = (203, 213, 225)


def font(size, bold=False):
    cands = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
             "C:/Windows/Fonts/msyh.ttc"]
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def rbox(d, xy, fill, outline, r=14, w=2):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=w)


def ctext(d, xy, text, f, fill=INK, anchor="mm", spacing=6):
    d.multiline_text(xy, text, font=f, fill=fill, anchor=anchor,
                     align="center", spacing=spacing)


def arrow(d, p1, p2, color=MUTED, w=3):
    d.line([p1, p2], fill=color, width=w)
    (x2, y2) = p2
    (x1, y1) = p1
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    L, sp = 13, 0.42
    p = [(x2, y2),
         (x2 - L * math.cos(ang - sp), y2 - L * math.sin(ang - sp)),
         (x2 - L * math.cos(ang + sp), y2 - L * math.sin(ang + sp))]
    d.polygon(p, fill=color)


def fig_flow():
    W, H = 2040, 780
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_t, f_b = font(40, True), font(26, True)
    f_s, f_xs = font(22), font(18)

    d.text((60, 40), "算法流程：一个输入，两条互补通道，一个代价最优判定", font=f_t, fill=INK)
    d.text((60, 96), "两条通道的失效模式不重叠 —— 一条管“细”，一条管“弱”", font=f_s, fill=MUTED)

    def box(x0, x1, y0, y1, fill, line, title, body, note="", tc=INK):
        rbox(d, (x0, y0, x1, y1), fill, line)
        cx = (x0 + x1) / 2
        h = y1 - y0
        ctext(d, (cx, y0 + h * 0.22), title, f_b, tc)
        ctext(d, (cx, y0 + h * 0.53), body, f_xs, INK)
        if note:
            ctext(d, (cx, y0 + h * 0.83), note, f_xs, MUTED)

    box(60, 250, 330, 460, GRAY_L, LINE, "输入图像", "640×480\n灰度")
    arrow(d, (250, 395), (300, 395))

    box(300, 540, 322, 468, BLUE_L, BLUE, "C1  光照归一化",
        "81×81 中值背景估计\n差分后零均值", "绝对灰度 → 相对反差", BLUE_D)

    d.line([(540, 395), (586, 395)], fill=MUTED, width=3)
    d.line([(586, 205), (586, 585)], fill=MUTED, width=3)
    arrow(d, (586, 205), (636, 205))
    arrow(d, (586, 585), (636, 585))

    box(636, 1000, 120, 290, AMBER_L, AMBER, "通道 A · 高频",
        "多尺度黑帽 7/15/25 取最大\n阈值 = mean + k·MADσ", "管“细”：1~3 px 划痕、凹坑", (146, 64, 14))
    box(636, 1000, 500, 670, GREEN_L, GREEN, "通道 B · 低频",
        "残差低通 σ=2.5，取 -k·MADσ\n把纹理噪声按 √N 压下去", "管“弱”：低对比污渍、油污", (6, 95, 70))

    arrow(d, (1000, 205), (1046, 205))
    arrow(d, (1000, 585), (1046, 585))
    d.line([(1046, 205), (1046, 585)], fill=MUTED, width=3)
    arrow(d, (1046, 395), (1086, 395))

    box(1086, 1370, 322, 468, BLUE_L, BLUE, "C2  并集 + 形态学",
        "两通道取并集\n4 方向各向异性闭运算", "圆核开运算会抹掉细划痕", BLUE_D)
    arrow(d, (1370, 395), (1410, 395))

    box(1410, 1694, 322, 468, BLUE_L, BLUE, "C3  评分与分级",
        "对比度 .5 / 面积 .3 / 形状 .2\n按面积定级：严重·中等·轻微", "", BLUE_D)
    arrow(d, (1694, 395), (1734, 395))

    box(1734, 1980, 322, 468, (254, 226, 226), (220, 38, 38), "代价最优判定",
        "10·FN + 1·FP 取最小", "输出 OK / NG + 明细报表", (153, 27, 27))

    y = 700
    d.line([(60, y), (W - 60, y)], fill=LINE, width=2)
    d.text((60, y + 26),
           "为什么两条通道都要有：去掉低频通道，油污召回从 96.6% 掉到 27.6%；去掉多尺度，划痕召回从 71% 掉到 39%。",
           font=f_xs, fill=MUTED)
    img.save(OUT / "fig_flow.png")


def fig_judge():
    """用真实阈值扫描结果画代价曲线（不画示意图）。"""
    import csv as _csv
    rows = []
    keys = {"score_thr": "score_thr", "精确率": "precision", "召回率": "recall",
            "漏检数": "fn", "误检数": "fp", "代价": "cost"}
    with (OUT / "tradeoff.csv").open(encoding="utf-8-sig") as fh:
        for r in _csv.DictReader(fh):
            rows.append({keys.get(k, k): float(v) for k, v in r.items()})
    if not rows:
        return

    W, H = 1400, 620
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_t, f_s, f_b = font(34, True), font(20), font(22, True)
    f_xs = font(17)
    d.text((60, 38), "判定阈值不是“越高越好”：代价曲线的最小值在偏左侧", font=f_t, fill=INK)

    L, R, T, B = 120, W - 110, 130, H - 130
    d.line([(L, T), (L, B)], fill=LINE, width=2)
    d.line([(L, B), (R, B)], fill=LINE, width=2)

    xs = [r["score_thr"] for r in rows]
    costs = [r["cost"] for r in rows]
    fns = [r["fn"] * 10 for r in rows]     # 换算成代价量纲，便于同图比较
    fps_ = [r["fp"] for r in rows]
    vmax = max(max(costs), max(fns)) * 1.15
    x0, x1 = min(xs), max(xs)

    def px(v):
        return L + (R - L) * (v - x0) / (x1 - x0)

    def py(v):
        return B - (B - T) * v / vmax

    for i in range(5):
        y = B - (B - T) * i / 4
        d.line([(L, y), (R, y)], fill=(238, 241, 245))
        d.text((L - 62, y - 9), f"{vmax * i / 4:.0f}", font=f_xs, fill=(140, 150, 165))
    for v in xs:
        d.text((px(v) - 14, B + 10), f"{v:.2f}", font=f_xs, fill=(140, 150, 165))
    d.text(((L + R) / 2 - 50, B + 38), "置信度判定阈值 score_thr", font=f_s, fill=MUTED)
    d.text((L - 74, T - 30), "代价 / 10·FN", font=f_s, fill=MUTED)

    d.line([(px(v), py(c)) for v, c in zip(xs, costs)], fill=(30, 64, 175), width=4)
    d.line([(px(v), py(c)) for v, c in zip(xs, fns)], fill=(217, 83, 79), width=2)
    d.line([(px(v), py(c)) for v, c in zip(xs, fps_)], fill=(16, 145, 96), width=2)

    def dot(i, color):
        x, y = px(xs[i]), py(costs[i])
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=color)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill="white")

    ic = min(range(len(costs)), key=lambda i: costs[i])
    i60 = min(range(len(costs)), key=lambda i: abs(xs[i] - 0.60))
    dot(i60, (217, 83, 79))
    dot(ic, (30, 64, 175))

    # 固定位置的说明框，避免压曲线
    bx, by, bw, bh = L + 10, T + 4, 336, 100
    d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=8, fill="white",
                        outline=(214, 220, 229), width=2)
    d.ellipse([bx + 16, by + 16, bx + 28, by + 28], fill=(30, 64, 175))
    d.text((bx + 38, by + 8), f"代价最优  {xs[ic]:.2f}", font=f_b, fill=(30, 64, 175))
    d.text((bx + 38, by + 36),
           f"漏检 {int(rows[ic]['fn'])} · 误检 {int(rows[ic]['fp'])} · 代价 {costs[ic]:.0f}",
           font=f_xs, fill=MUTED)
    d.ellipse([bx + 16, by + 64, bx + 28, by + 76], fill=(217, 83, 79))
    d.text((bx + 38, by + 56), f"F1 最优  {xs[i60]:.2f}", font=f_b, fill=(217, 83, 79))
    d.text((bx + 38, by + 84),
           f"漏检 {int(rows[i60]['fn'])} · 误检 {int(rows[i60]['fp'])} · 代价 {costs[i60]:.0f}",
           font=f_xs, fill=MUTED)

    lx = L
    for name, color in (("加权代价 10·FN+FP", (30, 64, 175)), ("漏检项 10·FN", (217, 83, 79)),
                        ("误检项 FP", (16, 145, 96))):
        d.rectangle([lx, H - 62, lx + 16, H - 46], fill=color)
        d.text((lx + 22, H - 64), name, font=f_xs, fill=(80, 90, 105))
        lx += 40 + len(name) * 15

    d.text((60, H - 30), "曲线来自 out/tradeoff.csv 的实测扫描：阈值抬高能压掉误检，但漏检按 10 倍代价记账后总成本反而上升。",
           font=f_xs, fill=MUTED)
    img.save(OUT / "fig_judge.png")


if __name__ == "__main__":
    fig_flow()
    fig_judge()
    print("figs ->", OUT)
