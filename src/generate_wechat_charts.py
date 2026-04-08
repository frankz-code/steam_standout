"""Generate WeChat 公众号 optimized static chart & table images.

Reads the same data as 10_generate_report.py, then renders each chart/table
as a 1080px-wide JPG using matplotlib, following WeChat design best practices.
No titles on chart images (titles go in the article body text).
"""

import importlib
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
report = importlib.import_module("10_generate_report")

WECHAT_DIR = PROJECT_ROOT / "wechat_charts"
WECHAT_DIR.mkdir(exist_ok=True)

# ── WeChat matplotlib config ─────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": ["Hiragino Sans GB", "STHeiti", "Heiti TC", "sans-serif"],
    "font.size": 20,
    "axes.titlesize": 36,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "text.color": "#3f3f3f",
    "axes.labelcolor": "#3f3f3f",
    "axes.edgecolor": "#d1d5db",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "axes.grid": False,
    "legend.frameon": True,
    "legend.edgecolor": "#e5e7eb",
    "legend.facecolor": "#ffffff",
    "savefig.dpi": 100,
})

# ── Colors (matching report) ─────────────────────────────────────────
C = {
    "green": "#10b981",
    "gray": "#94a3b8",
    "blue": "#3b82f6",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "primary": "#1b2838",
    "text": "#3f3f3f",
    "light_bg": "#f8f9fa",
    "border": "#e5e7eb",
    "header": "#0f4c81",
}


def save(fig, name):
    """Save figure as JPG for WeChat at 1080px width."""
    path = WECHAT_DIR / f"{name}.jpg"
    fig.savefig(
        path, dpi=100, bbox_inches="tight", pad_inches=0.02,
        facecolor="white", format="jpg",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)
    print(f"  → {path}")


def fmt_k(v):
    """Format number: 20000 → '20K'."""
    if abs(v) >= 1000:
        return f"{v/1000:.0f}K"
    return f"{v:.0f}"


# ══════════════════════════════════════════════════════════════════════
#  CHART RENDERERS
# ══════════════════════════════════════════════════════════════════════

def chart_avg_vs_peak(peak_data_full):
    """Scatter: avg daily vs peak week velocity (log-log)."""
    gf = pd.DataFrame(peak_data_full)
    s = gf[gf["group"] == "success"]
    b = gf[gf["group"] != "success"]

    fig, ax = plt.subplots(figsize=(10.8, 7.5), dpi=100)

    # Backdrop scatter
    bx = b["avg_daily"].clip(lower=0.3)
    by = b["peak_week"].clip(lower=1)
    ax.scatter(bx, by, s=12, c=C["gray"], alpha=0.2, zorder=2,
               label="Backdrop样本", edgecolors="none")

    # Confidence ellipses
    for data, color, alpha, label in [
        (b, C["gray"], 0.08, None),
        (s, C["green"], 0.12, None),
    ]:
        pts = data[(data["avg_daily"] > 0) & (data["peak_week"] > 0)]
        if len(pts) < 3:
            continue
        lx = np.log10(pts["avg_daily"].clip(lower=0.3).values)
        ly = np.log10(pts["peak_week"].values)
        mx, my = lx.mean(), ly.mean()
        cov = np.cov(lx, ly)
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        w, h = 2 * 2 * np.sqrt(vals)  # 2-sigma
        ell = Ellipse(
            (10**mx, 10**my), 10**(mx + w / 2) - 10**(mx - w / 2),
            10**(my + h / 2) - 10**(my - h / 2),
            angle=angle, facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=1.5, linestyle="--", zorder=1,
        )
        # Use manual path for log-scale ellipse
        theta = np.linspace(0, 2 * np.pi, 80)
        a, b_ax = 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0])
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        ex = a * np.cos(theta) * cos_a - b_ax * np.sin(theta) * sin_a + mx
        ey = a * np.cos(theta) * sin_a + b_ax * np.sin(theta) * cos_a + my
        ax.fill(10**ex, 10**ey, color=color, alpha=alpha, zorder=1)
        ax.plot(10**ex, 10**ey, color=color, linewidth=1.5, linestyle="--",
                alpha=0.5, zorder=1)

    # Success scatter
    sx = s["avg_daily"].clip(lower=0.3)
    sy = s["peak_week"].clip(lower=1)
    ax.scatter(sx, sy, s=80, c=C["green"], alpha=0.85, zorder=4,
               label="爆款", edgecolors="white", linewidths=0.5)

    # Labels for success games
    for _, row in s.iterrows():
        if row["avg_daily"] > 0:
            ax.annotate(
                row["name"], (max(row["avg_daily"], 0.3), row["peak_week"]),
                fontsize=9, color="#065f46", fontweight="bold",
                xytext=(6, 2), textcoords="offset points",
                zorder=5,
            )

    # Threshold lines
    ax.axhline(900, color="#b0b8c4", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(40, color="#b0b8c4", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.35, 920, "巅峰周≥900", fontsize=16, color=C["gray"],
            ha="left", va="bottom")
    ax.text(42, 1.2, "日常≥40", fontsize=16, color=C["gray"],
            ha="left", va="bottom", rotation=90)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, None)
    ax.set_ylim(1, None)
    ax.set_xlabel("日常增速（中位数，人/天）", fontsize=22)
    ax.set_ylabel("巅峰周增速（人/天）", fontsize=22)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
    ax.legend(loc="lower right", fontsize=20, markerscale=1.5)
    ax.grid(True, which="major", alpha=0.12, color="#94a3b8")
    fig.tight_layout()
    save(fig, "chart_01_avg_vs_peak")


def chart_trajectories(trajectories, control_envelope):
    """Line: follower growth trajectories vs control median."""
    fig, ax = plt.subplots(figsize=(10.8, 7.0), dpi=100)
    game_colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]

    # Control envelope
    ce = sorted(control_envelope, key=lambda p: p[0], reverse=True)
    cx = [p[0] for p in ce]
    cy = [p[1] for p in ce]
    ax.plot(cx, cy, color=C["gray"], linewidth=2, linestyle="--",
            label="对照组中位数", zorder=2)
    ax.fill_between(cx, cy, alpha=0.06, color=C["gray"])

    idx = 0
    for name, data in trajectories.items():
        data_sorted = sorted(data, key=lambda p: p[0], reverse=True)
        gx = [p[0] for p in data_sorted]
        gy = [p[1] for p in data_sorted]
        color = game_colors[idx % len(game_colors)]
        ax.plot(gx, gy, color=color, linewidth=2.5, label=name, zorder=3)
        # Direct label at end of line
        if gy:
            ax.annotate(name, (gx[-1], gy[-1]), fontsize=16, color=color,
                        fontweight="bold", xytext=(8, 0),
                        textcoords="offset points", va="center")
        idx += 1

    ax.set_xlim(200, -7)
    ax.set_xlabel("距上线天数", fontsize=22)
    ax.set_ylabel("关注者总数", fontsize=22)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"T-{int(v)}" if v > 0 else ("上线" if v == 0 else f"T+{int(abs(v))}")))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
    ax.legend(loc="upper right", fontsize=18)
    ax.grid(True, axis="y", alpha=0.15, color="#94a3b8")
    fig.tight_layout()
    save(fig, "chart_02_trajectories")


def chart_peak_annotated(peak_annotated):
    """Line: 7d rolling velocity trajectories with peak pins."""
    fig, ax = plt.subplots(figsize=(10.8, 7.0), dpi=100)
    color_map = {
        "漫威争锋": "#10b981", "ARC Raiders": "#3b82f6",
        "Manor Lords": "#f59e0b", "Schedule I": "#ef4444",
    }
    fallback = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    for i, game in enumerate(peak_annotated):
        clr = color_map.get(game["name"], fallback[i % 4])
        data = sorted(game["data"], key=lambda p: p[0], reverse=True)
        gx = [p[0] for p in data]
        gy = [p[1] for p in data]
        ax.plot(gx, gy, color=clr, linewidth=2.5, label=game["name"], zorder=3)

        # Mark peak with annotation
        if game.get("peak_dtl") and game.get("peak_week"):
            ax.plot(game["peak_dtl"], game["peak_week"], "v", color=clr,
                    markersize=14, zorder=5)
            ax.annotate(
                f'{int(game["peak_week"])}/天',
                (game["peak_dtl"], game["peak_week"]),
                fontsize=16, color=clr, fontweight="bold",
                xytext=(0, 14), textcoords="offset points",
                ha="center", va="bottom", zorder=5,
            )

    ax.set_xlim(300, 0)
    ax.set_xlabel("距上线天数", fontsize=22)
    ax.set_ylabel("7日滚动均增（人/天）", fontsize=22)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"T-{int(v)}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
    ax.legend(loc="upper left", fontsize=18)
    ax.grid(True, axis="y", alpha=0.15, color="#94a3b8")
    fig.tight_layout()
    save(fig, "chart_03_peak_annotated")


def chart_windows(velocity_windows):
    """Grouped bar: velocity by time window, success vs control."""
    windows = ["T-360→T-180", "T-180→T-90", "T-90→T-30", "T-30→上线"]
    s_vals, c_vals = [], []
    for w in windows:
        sd = next((d for d in velocity_windows if d["window"] == w and d["group"] == "success"), None)
        cd = next((d for d in velocity_windows if d["window"] == w and d["group"] == "control"), None)
        s_vals.append(sd["median"] if sd else 0)
        c_vals.append(cd["median"] if cd else 0)

    ratios = [f"{s / c:.1f}x" if c > 0 else "-" for s, c in zip(s_vals, c_vals)]

    fig, ax = plt.subplots(figsize=(10.8, 6.5), dpi=100)
    x = np.arange(len(windows))
    width = 0.35
    bars1 = ax.bar(x - width / 2, s_vals, width, color=C["green"], label="爆款", zorder=3)
    bars2 = ax.bar(x + width / 2, c_vals, width, color=C["gray"], label="对照组", zorder=3)

    # Data labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{int(bar.get_height())}", ha="center", va="bottom",
                fontsize=18, fontweight="bold", color=C["green"])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{int(bar.get_height())}", ha="center", va="bottom",
                fontsize=18, color=C["gray"])

    # Ratio annotations at top
    y_max = max(s_vals) * 1.25
    for i, r in enumerate(ratios):
        ax.text(x[i], y_max, f"{r}倍", ha="center", va="bottom",
                fontsize=20, fontweight="bold", color=C["red"])

    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=20)
    ax.set_ylabel("中位增速（人/天）", fontsize=22)
    ax.legend(fontsize=20, loc="upper left")
    ax.set_ylim(0, y_max * 1.15)
    ax.grid(True, axis="y", alpha=0.12, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "chart_04_windows")


def chart_threshold(threshold_sweep):
    """Line: precision/recall vs peak threshold."""
    fig, ax = plt.subplots(figsize=(10.8, 6.5), dpi=100)
    thresholds = [d["threshold"] for d in threshold_sweep]
    precision = [d["precision"] for d in threshold_sweep]
    recall = [d["recall"] for d in threshold_sweep]

    ax.plot(thresholds, precision, color=C["green"], linewidth=2.5,
            label="命中率", zorder=3)
    ax.plot(thresholds, recall, color=C["blue"], linewidth=2.5,
            label="不漏球率", zorder=3)

    # Strategy threshold markers
    markers = [
        (400, "广角 400", C["green"]),
        (900, "精选 900", C["amber"]),
        (1900, "高确信 1900", C["red"]),
    ]
    for thresh, label, color in markers:
        ax.axvline(thresh, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
        # Find precision at this threshold
        prec_val = None
        for d in threshold_sweep:
            if d["threshold"] >= thresh:
                prec_val = d["precision"]
                break
        y_pos = 0.92 if thresh == 400 else (0.85 if thresh == 900 else 0.78)
        ax.text(thresh * 1.08, y_pos, f"{label}\n命中{prec_val:.0%}" if prec_val else label,
                fontsize=16, color=color, fontweight="bold", va="top")

    ax.set_xscale("log")
    ax.set_xlim(50, 10000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("巅峰周增速门槛（人/天）", fontsize=22)
    ax.set_ylabel("比率", fontsize=22)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=20, loc="center left")
    ax.grid(True, alpha=0.12, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "chart_05_threshold")


def chart_strategy_quality(strategy_quality):
    """Grouped bar: $10M%, $50M%, Top100% by strategy."""
    tiers = strategy_quality["tiers"]
    labels = [t["label"] for t in tiers]

    fig, ax = plt.subplots(figsize=(10.8, 6.5), dpi=100)
    x = np.arange(len(labels))
    width = 0.25

    metrics = [
        ("pct_10m", "收入>$10M占比", C["green"]),
        ("pct_50m", "收入>$50M占比", C["blue"]),
        ("pct_top100", "进入Top100占比", C["amber"]),
    ]

    for i, (key, label, color) in enumerate(metrics):
        vals = [t[key] for t in tiers]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=color,
                      label=label, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=18, loc="upper left")
    ax.grid(True, axis="y", alpha=0.12, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "chart_06_strategy_quality")


def chart_detection_box(detection_50m):
    """Boxplot: detection leadtime for $50M+ games."""
    tiers = detection_50m["tiers"]
    fig, ax = plt.subplots(figsize=(10.8, 7.0), dpi=100)

    positions = list(range(len(tiers)))
    tier_colors = [t["color"] for t in tiers]

    # Draw boxplots
    bp_data = []
    for t in tiers:
        bp_data.append(t["values"])

    bp = ax.boxplot(
        bp_data, positions=positions, widths=0.4, patch_artist=True,
        showfliers=False, zorder=2,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color + "40")
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
    for i, (whisker, cap) in enumerate(zip(
            zip(bp["whiskers"][::2], bp["whiskers"][1::2]),
            zip(bp["caps"][::2], bp["caps"][1::2]))):
        color = tier_colors[i // 1] if i < len(tier_colors) else C["gray"]
    for i in range(len(tiers)):
        color = tier_colors[i]
        bp["whiskers"][2 * i].set_color(color)
        bp["whiskers"][2 * i + 1].set_color(color)
        bp["caps"][2 * i].set_color(color)
        bp["caps"][2 * i + 1].set_color(color)

    # Scatter overlay with jitter
    for i, t in enumerate(tiers):
        jitter = np.random.uniform(-0.12, 0.12, len(t["values"]))
        ax.scatter(
            [i + j for j in jitter], t["values"],
            s=30, c=t["color"], alpha=0.6, zorder=3, edgecolors="none",
        )

    # Reference lines
    for days, label in [(30, "30天"), (90, "90天"), (270, "270天")]:
        ax.axhline(days, color=C["amber"], linestyle="--", linewidth=1, alpha=0.7)
        ax.text(len(tiers) - 0.5, days + 5, label, fontsize=18,
                color=C["amber"], fontweight="bold", ha="right", va="bottom")

    # Summary cards above each boxplot
    y_top = ax.get_ylim()[1]
    for i, t in enumerate(tiers):
        ax.text(i, y_top * 1.02,
                f'{t["detected"]}/{t["total"]}款命中',
                ha="center", va="bottom", fontsize=18,
                fontweight="bold", color=t["color"])
        ax.text(i, y_top * 0.94,
                f'中位 {int(t["median"])}天 | 均值 {int(t["mean"])}天',
                ha="center", va="bottom", fontsize=16,
                fontweight="bold", color=C["text"])

    ax.set_xticks(positions)
    ax.set_xticklabels([t["label"] for t in tiers], fontsize=20, fontweight="bold")
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(tier_colors[i])
    ax.set_ylabel("提前发现天数", fontsize=22)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v)}天"))
    ax.grid(True, axis="y", alpha=0.1, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "chart_07_detection_box")


# ══════════════════════════════════════════════════════════════════════
#  TABLE RENDERERS
# ══════════════════════════════════════════════════════════════════════

def _draw_table(headers, rows, col_widths, col_aligns, filename,
                header_fontsize=18, body_fontsize=16, row_height_px=52,
                header_height_px=56, bold_first_col=True):
    """Render a table as a clean WeChat-optimized image using pixel coords.

    col_widths: list of floats summing to 1.0 (relative widths).
    col_aligns: list of 'left', 'center', 'right' per column.
    """
    n_rows = len(rows)
    W = 1080  # figure width in px
    margin_px = 0   # no outer margin — edge-to-edge
    pad_top = 0
    pad_bottom = 0
    pad_x = 10  # cell text padding

    table_w = W - 2 * margin_px
    total_h = pad_top + header_height_px + n_rows * row_height_px + pad_bottom

    fig_w = W / 100  # inches at 100 dpi
    fig_h = total_h / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(total_h, 0)  # y=0 at top

    # Column x boundaries
    cum_x = [margin_px]
    for w in col_widths:
        cum_x.append(cum_x[-1] + w * table_w)

    # ── Header ──
    y0 = pad_top
    ax.add_patch(Rectangle(
        (margin_px, y0), table_w, header_height_px,
        facecolor=C["header"], edgecolor="none", zorder=2,
    ))
    for j, (hdr, align) in enumerate(zip(headers, col_aligns)):
        if align == "right":
            tx = cum_x[j + 1] - pad_x
            ha = "right"
        elif align == "center":
            tx = (cum_x[j] + cum_x[j + 1]) / 2
            ha = "center"
        else:
            tx = cum_x[j] + pad_x
            ha = "left"
        ax.text(tx, y0 + header_height_px / 2, hdr, ha=ha, va="center",
                fontsize=header_fontsize, fontweight="bold", color="white")

    # ── Body rows ──
    for i, row in enumerate(rows):
        ry = pad_top + header_height_px + i * row_height_px
        bg = C["light_bg"] if i % 2 == 0 else "white"
        ax.add_patch(Rectangle(
            (margin_px, ry), table_w, row_height_px,
            facecolor=bg, edgecolor=C["border"], linewidth=0.5, zorder=1,
        ))
        for j, (cell, align) in enumerate(zip(row, col_aligns)):
            if align == "right":
                tx = cum_x[j + 1] - pad_x
                ha = "right"
            elif align == "center":
                tx = (cum_x[j] + cum_x[j + 1]) / 2
                ha = "center"
            else:
                tx = cum_x[j] + pad_x
                ha = "left"
            fw = "bold" if (j == 0 and bold_first_col) else "normal"
            ax.text(tx, ry + row_height_px / 2, str(cell), ha=ha, va="center",
                    fontsize=body_fontsize, fontweight=fw, color=C["text"])

    save(fig, filename)


def table_success_games(success_games):
    """Table 1: 30 success games."""
    headers = ["游戏", "累计销量（万份）", "发行商", "发售日"]
    col_widths = [0.28, 0.18, 0.32, 0.22]
    col_aligns = ["left", "right", "left", "center"]

    rows = []
    for g in success_games[:30]:
        if g["copies_sold"] >= 1e4:
            copies = f'{g["copies_sold"] / 1e4:.0f}'
        elif g["copies_sold"] > 0:
            copies = f'{g["copies_sold"]:,}'
        else:
            copies = "—"
        pub = g["publisher"]
        if len(pub) > 20:
            pub = pub[:18] + "…"
        rows.append([g["name"], copies, pub, g["launch"]])

    _draw_table(headers, rows, col_widths, col_aligns,
                "table_01_success_games", header_fontsize=16, body_fontsize=14,
                row_height_px=42, header_height_px=50)


def table_signal_strength(peak_d_comparison):
    """Table 2: Peak/avg signal strength comparison."""
    d = peak_d_comparison
    headers = ["指标", "爆款中位数", "对照中位数", "倍率", "Cohen's d"]
    col_widths = [0.22, 0.22, 0.20, 0.16, 0.20]
    col_aligns = ["left", "right", "right", "right", "right"]
    rows = [
        ["巅峰周增速", f'{int(d["peak_s_median"])}/天',
         f'{int(d["peak_b_median"])}/天',
         f'{d["peak_ratio"]}x', f'{d["peak_week_d"]}'],
        ["日常增速（中位数）", f'{int(d["avg_s_median"])}/天',
         f'{int(d["avg_b_median"])}/天',
         f'{d["avg_ratio"]}x', f'{d["avg_daily_d"]}'],
    ]
    _draw_table(headers, rows, col_widths, col_aligns, "table_02_signal_strength")


def table_strategy_tiers(tiers):
    """Table 3: Three strategy tiers."""
    headers = ["策略", "巅峰周≥", "日常≥", "关注数", "命中率", "不漏球率"]
    col_widths = [0.22, 0.16, 0.14, 0.16, 0.16, 0.16]
    col_aligns = ["left", "right", "right", "right", "right", "right"]
    rows = []
    for t in tiers:
        rows.append([
            t["cn_label"],
            f'{t["peak_threshold"]}/天',
            f'{t["avg_threshold"]}/天',
            f'{t["radar"]}款',
            f'{t["precision"]:.1%}',
            f'{t["recall"]:.0%}',
        ])
    _draw_table(headers, rows, col_widths, col_aligns, "table_03_strategy_tiers")


def table_strategy_quality_detail(strategy_quality):
    """Table 4: Strategy quality with $50M metrics."""
    sq = strategy_quality
    headers = ["策略", "命中/关注", "命中率", "不漏球率", "看走眼", "收入中位数", "Top100"]
    col_widths = [0.16, 0.13, 0.12, 0.13, 0.12, 0.17, 0.17]
    col_aligns = ["left", "right", "right", "right", "right", "right", "right"]
    rows = []
    for t in sq["tiers"]:
        rows.append([
            t["label"],
            f'{t["n_50m"]}/{t["n"]}',
            f'{t["hit_50m_precision"]:.0%}',
            f'{t["hit_50m_recall"]:.0%}',
            f'{t["fps_50m"]}款',
            f'${t["median_rev"]}M',
            f'{t["n_top100"]}/{t["n"]}',
        ])
    _draw_table(headers, rows, col_widths, col_aligns,
                "table_04_strategy_quality", header_fontsize=16, body_fontsize=15)


def table_roi(strategy_quality):
    """Table 5: ROI — games per $10M+ discovery."""
    headers = ["策略", "关注数", "$10M+产品", "每发现1个需看"]
    col_widths = [0.28, 0.22, 0.22, 0.28]
    col_aligns = ["left", "right", "right", "right"]
    rows = []
    for t in strategy_quality["tiers"]:
        rows.append([
            t["label"],
            f'{t["n"]}款',
            f'{t["n_10m"]}款',
            f'{t["games_per_10m"]}款',
        ])
    _draw_table(headers, rows, col_widths, col_aligns, "table_05_roi")


def table_stealth():
    """Table 6: Stealth games the strategy can't detect."""
    headers = ["游戏", "巅峰周增速", "日常增速中位数", "首月营收"]
    col_widths = [0.25, 0.22, 0.25, 0.28]
    col_aligns = ["left", "right", "right", "right"]
    rows = [
        ["致命公司", "57/天", "2/天", "$146M"],
        ["恶魔轮盘", "269/天", "152/天", "$17M"],
        ["小丑牌", "214/天", "26/天", "$78M"],
        ["Megabonk", "87/天", "8/天", "$32M"],
    ]
    _draw_table(headers, rows, col_widths, col_aligns, "table_06_stealth")


def table_no_difference():
    """Table 7: Indicators with no signal."""
    headers = ["指标", "Cohen's d", "意味着什么"]
    col_widths = [0.25, 0.15, 0.60]
    col_aligns = ["left", "right", "left"]
    rows = [
        ["售价", "0.09", "什么价格都能爆"],
        ["开发者历史营收", "0.01", "新工作室和老工作室一样可能出爆款"],
        ["上线前挂页天数", "0.08", "挂半年和挂两年都行"],
        ["标签数量", "0.25", "给自己打满标签也用处不大"],
    ]
    _draw_table(headers, rows, col_widths, col_aligns, "table_07_no_difference")


# ══════════════════════════════════════════════════════════════════════
#  STRATEGY PANELS (static 3-column view of each tier)
# ══════════════════════════════════════════════════════════════════════

def _vel_short(v):
    """Format velocity compactly: 19450 → '19K', 269 → '269'."""
    if v >= 10000:
        return f"{v/1000:.0f}K"
    elif v >= 1000:
        return f"{v/1000:.1f}K".replace(".0K", "K")
    return str(v)


def _clip_name(name, max_chars=14):
    """Truncate a game name if too long for the column."""
    if len(name) <= max_chars:
        return name
    return name[:max_chars - 1] + "…"


def strategy_panel(tier, filename):
    """Render one strategy tier as a 3-column panel image (命中 / 遗漏 / 看走眼).

    Mimics the interactive HTML panel: summary stats at top, then 3 columns
    of game names with velocities. Max 16 visible items per column.
    """
    W = 1080
    margin = 0           # no outer margin — edge-to-edge
    content_w = W - 2 * margin

    # ── Layout constants ──
    stat_h = 80          # summary stats bar height
    col_hdr_h = 36       # column header height
    row_h = 30           # per-game row height
    max_rows = 16        # max visible items per column (incl "...及其他")
    gap = 4              # between stats bar and columns
    pad_top = 0
    pad_bottom = 0
    col_gap = 2          # thin gap between columns
    name_font = 11.5
    vel_font = 10.5
    vel_zone = 52        # px reserved for velocity text on the right

    # ── Prepare column data ──
    def build_items(entries, is_miss=False, no_data=None):
        """Build display items: list of (name, vel_str, is_divider, style)."""
        items = []
        for e in entries:
            vel = int(e["peak"]) if e["peak"] > 0 else 0
            vel_str = f"{_vel_short(vel)}/天" if vel > 0 else "—"
            items.append((e["name"], vel_str, False, "normal"))

        if is_miss and no_data:
            items.append(("── 无上线前数据 ──", "", True, "divider"))
            for nm in no_data:
                items.append((nm, "—", False, "nodata"))

        total = len(items)
        if total > max_rows:
            shown = items[:max_rows - 1]
            remaining = total - (max_rows - 1)
            shown.append((f"… 及其他 {remaining} 款", "", False, "more"))
            return shown, total
        return items, total

    hits_items, hits_total = build_items(tier["hits"])
    miss_items, miss_total = build_items(
        tier["misses"], is_miss=True, no_data=tier["no_data"])
    fps_items, fps_total = build_items(tier["fps"])

    # Number of body rows = max across 3 columns
    body_rows = max(len(hits_items), len(miss_items), len(fps_items))
    body_rows = max(body_rows, 1)

    total_h = pad_top + stat_h + gap + col_hdr_h + body_rows * row_h + pad_bottom

    fig_w = W / 100
    fig_h = total_h / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(total_h, 0)  # y=0 at top

    # ── Draw summary stats bar ──
    stat_y = pad_top
    ax.add_patch(Rectangle(
        (margin, stat_y), content_w, stat_h,
        facecolor="#f1f5f9", edgecolor=C["border"], linewidth=0.5, zorder=1,
    ))

    # 3 stats aligned with 3 columns below
    col_w = (content_w - 2 * col_gap) / 3
    stats = [
        (f"{tier['hit_count']}/关注{tier['radar']}", f"命中率 {tier['precision']*100:.1f}%", C["green"]),
        (f"{tier['hit_count']}/{tier['evaluable_total']}", f"不漏球率 {tier['recall']*100:.0f}%", C["red"]),
        (str(tier['fp_count']), "看走眼？", C["amber"]),
    ]
    for i, (val, label, color) in enumerate(stats):
        cx = margin + i * (col_w + col_gap) + col_w / 2
        cy_val = stat_y + stat_h * 0.33
        cy_lbl = stat_y + stat_h * 0.72
        ax.text(cx, cy_val, val, ha="center", va="center",
                fontsize=24, fontweight="bold", color=color, zorder=2)
        ax.text(cx, cy_lbl, label, ha="center", va="center",
                fontsize=12, color="#94a3b8", zorder=2)

    # ── Column definitions ──
    columns = [
        (f"命中（{tier['hit_count']}）", C["green"], hits_items),
        (f"遗漏（{tier['miss_count']} + {tier['no_data_count']}无数据）",
         C["red"], miss_items),
        (f"看走眼？（{tier['fp_count']}）", C["amber"], fps_items),
    ]

    col_top = stat_y + stat_h + gap

    for ci, (header, hdr_color, items) in enumerate(columns):
        cx0 = margin + ci * (col_w + col_gap)

        # Column header
        ax.add_patch(Rectangle(
            (cx0, col_top), col_w, col_hdr_h,
            facecolor=hdr_color, edgecolor="none", zorder=2,
        ))
        ax.text(cx0 + col_w / 2, col_top + col_hdr_h / 2, header,
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white", zorder=3)

        # Column body rows
        for ri in range(body_rows):
            ry = col_top + col_hdr_h + ri * row_h
            bg = "#f8f9fa" if ri % 2 == 0 else "white"
            ax.add_patch(Rectangle(
                (cx0, ry), col_w, row_h,
                facecolor=bg, edgecolor="#e5e7eb", linewidth=0.3, zorder=1,
            ))

            if ri < len(items):
                name, vel, is_div, style = items[ri]
                if is_div:
                    ax.text(cx0 + col_w / 2, ry + row_h / 2, name,
                            ha="center", va="center", fontsize=10,
                            color="#94a3b8", zorder=2)
                elif style == "more":
                    ax.text(cx0 + 8, ry + row_h / 2, name,
                            ha="left", va="center", fontsize=10.5,
                            color="#94a3b8", style="italic", zorder=2)
                elif style == "nodata":
                    ax.text(cx0 + 8, ry + row_h / 2, _clip_name(name),
                            ha="left", va="center", fontsize=name_font,
                            color="#94a3b8", zorder=2)
                    ax.text(cx0 + col_w - 8, ry + row_h / 2, vel,
                            ha="right", va="center", fontsize=vel_font,
                            color="#c4c4c4", zorder=2)
                else:
                    ax.text(cx0 + 8, ry + row_h / 2, _clip_name(name),
                            ha="left", va="center", fontsize=name_font,
                            color="#1e293b", zorder=2,
                            clip_on=True)
                    ax.text(cx0 + col_w - 8, ry + row_h / 2, vel,
                            ha="right", va="center", fontsize=vel_font,
                            color="#94a3b8", zorder=2)

    save(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("Loading data (reusing 10_generate_report.py functions)...")
    af, fwd, bd, success_names, name_lookup = report.load_all_data()
    success_ids = set(success_names.keys())

    print("Extracting all report data...")
    success_games = report.extract_success_games_table(af, bd, success_names)

    peak_data = report.extract_peak_week_data(fwd, name_lookup)
    peak_data_full = report.extract_peak_week_from_histories(success_ids, name_lookup)

    # Trajectories (same showcase games as report)
    showcase_ids = ["3164500", "3241660", "2379780"]
    trajectories = report.extract_trajectories(fwd, showcase_ids, name_lookup)
    control_envelope = report.extract_control_envelope(fwd)

    # Peak annotated (same games as report)
    peak_annotated_ids = ["2767030", "1808500", "1363080", "3164500"]
    peak_annotated = report.extract_trajectory_with_peak(
        fwd, peak_annotated_ids, name_lookup, peak_data)

    velocity_windows = report.extract_velocity_windows(fwd)
    threshold_sweep = report.extract_threshold_sweep(peak_data_full, avg_floor=20)
    tiers = report.extract_strategy_tiers(peak_data_full, name_lookup, success_names)
    strategy_quality = report.extract_strategy_quality(peak_data_full, bd=bd)
    detection_50m = report.extract_detection_leadtime_50m(
        peak_data_full, name_lookup=name_lookup)
    peak_d_comparison = report.extract_peak_cohens_d(peak_data, peak_data_full)

    # ── Generate Charts ──
    print("\n=== Generating WeChat Charts ===")

    print("Chart 1: Avg vs Peak scatter...")
    chart_avg_vs_peak(peak_data_full)

    print("Chart 2: Trajectories...")
    chart_trajectories(trajectories, control_envelope)

    print("Chart 3: Peak annotated...")
    chart_peak_annotated(peak_annotated)

    print("Chart 4: Velocity windows...")
    chart_windows(velocity_windows)

    print("Chart 5: Threshold sweep...")
    chart_threshold(threshold_sweep)

    print("Chart 6: Strategy quality...")
    chart_strategy_quality(strategy_quality)

    print("Chart 7: Detection boxplot...")
    chart_detection_box(detection_50m)

    # ── Generate Tables ──
    print("\n=== Generating WeChat Tables ===")

    print("Table 1: Success games...")
    table_success_games(success_games)

    print("Table 2: Signal strength...")
    table_signal_strength(peak_d_comparison)

    print("Table 3: Strategy tiers...")
    table_strategy_tiers(tiers)

    print("Table 4: Strategy quality...")
    table_strategy_quality_detail(strategy_quality)

    print("Table 5: ROI...")
    table_roi(strategy_quality)

    print("Table 6: Stealth games...")
    table_stealth()

    print("Table 7: No difference indicators...")
    table_no_difference()

    # ── Generate Strategy Panels ──
    print("\n=== Generating Strategy Panels ===")
    tier_names = ["广角策略", "精选策略", "高确信策略"]
    for i, tier in enumerate(tiers):
        label = tier_names[i]
        fname = f"panel_{i+1}_{['wide', 'curated', 'conviction'][i]}"
        print(f"Panel {i+1}: {label}...")
        strategy_panel(tier, fname)

    print(f"\nDone! All images saved to {WECHAT_DIR}")


if __name__ == "__main__":
    main()
