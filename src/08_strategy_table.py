"""Generate a visual strategy table for the 4-tier threshold screening."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FORWARD_FILE = DATA_DIR / "forward_observations.parquet"
CONFIG_DIR = PROJECT_ROOT / "config"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"


def build_per_game_features(fwd):
    fl = fwd[(fwd["metric"] == "followers") & (fwd["days_to_launch"] > 0)].copy()
    games = []
    for sid, g in fl.groupby("steamId"):
        group = g["group"].iloc[0]
        has_onset = g["onset_artifact"].iloc[0] if "onset_artifact" in g.columns else False
        rates = g["daily_growth_7d"].dropna()
        if has_onset and len(rates) > 1:
            rates = rates.iloc[1:]
        if rates.empty:
            continue
        games.append({
            "steamId": sid,
            "group": group,
            "peak_week_velocity": rates.max(),
        })
    return pd.DataFrame(games)


def shorten(name, maxlen=20):
    name = str(name)
    return name[:maxlen-1] + "…" if len(name) > maxlen else name


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fwd = pd.read_parquet(FORWARD_FILE)
    success_list = json.loads(SUCCESS_FILE.read_text())
    success_names = {str(g["steamId"]): g["name"] for g in success_list}
    all_success_ids = set(success_names.keys())

    bd = pd.read_parquet(BACKDROP_FILE)
    bd["steamId"] = bd["steamId"].astype(str)
    name_lookup = dict(zip(bd["steamId"], bd["name"]))
    name_lookup.update(success_names)

    gf = build_per_game_features(fwd)
    evaluable_success = set(gf[gf["group"] == "success"]["steamId"])
    not_evaluable = all_success_ids - evaluable_success

    tiers = [
        ("Wide Net", 448),
        ("Sweet Spot", 845),
        ("Tighter List", 1626),
        ("High Conviction", 4140),
    ]

    # Build data
    tier_data = []
    for label, threshold in tiers:
        flagged = gf[gf["peak_week_velocity"] >= threshold].sort_values("peak_week_velocity", ascending=False)
        hits = flagged[flagged["group"] == "success"]
        fps = flagged[flagged["group"] == "control"]
        missed = gf[(gf["group"] == "success") & (gf["peak_week_velocity"] < threshold)]
        not_eval_names = sorted([success_names.get(sid, sid) for sid in not_evaluable])

        tier_data.append({
            "label": label,
            "threshold": threshold,
            "radar": len(flagged),
            "hit_rate": f"{len(hits)}/{len(flagged)} = {len(hits)/len(flagged):.0%}" if len(flagged) > 0 else "—",
            "catch_rate": f"{len(hits)}/{len(evaluable_success)} = {len(hits)/len(evaluable_success):.0%}",
            "watchlist": [name_lookup.get(r["steamId"], r["steamId"]) for _, r in flagged.iterrows()],
            "hits": [name_lookup.get(r["steamId"], r["steamId"]) for _, r in hits.iterrows()],
            "misses": [name_lookup.get(r["steamId"], r["steamId"]) for _, r in missed.iterrows()],
            "misses_no_data": not_eval_names,
            "fps": [name_lookup.get(r["steamId"], r["steamId"]) for _, r in fps.iterrows()],
        })

    # --- Render one table per tier as a 4-panel figure ---
    fig, axes = plt.subplots(4, 1, figsize=(22, 36))

    tier_bg = ["#f0faf0", "#e8f5e8", "#fff5eb", "#fde8e8"]

    for idx, (t, ax) in enumerate(zip(tier_data, axes)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor(tier_bg[idx])

        # Title bar
        title_colors = ["#27ae60", "#1e8449", "#e67e22", "#c0392b"]
        ax.add_patch(plt.Rectangle((0, 9.2), 10, 0.8, fc=title_colors[idx], ec="none"))
        ax.text(0.15, 9.6, f"{t['label']}    |    Threshold: ≥{t['threshold']}/day    |    "
                f"Radar: {t['radar']} games    |    Hit Rate: {t['hit_rate']}    |    "
                f"Catch Rate: {t['catch_rate']}",
                fontsize=11, fontweight="bold", color="white", va="center")

        # 4 columns below
        col_x = [0.1, 2.6, 5.1, 7.6]
        col_w = 2.3
        col_headers = ["Hits in Watchlist", "Missed in Success", "False Positives", "Full Watchlist"]
        col_colors = ["#27ae60", "#c0392b", "#d35400", "#2c3e50"]
        col_bg = ["#e8f8e8", "#fce8e8", "#fef3e2", "#f5f5f5"]
        col_data = [t["hits"], t["misses"], t["fps"], t["watchlist"]]
        col_extra = [None, t["misses_no_data"], None, None]

        for j in range(4):
            # Column header
            ax.add_patch(plt.Rectangle((col_x[j], 8.6), col_w, 0.5, fc=col_colors[j], ec="none", alpha=0.85))
            count = len(col_data[j])
            if j == 1 and col_extra[j]:
                count_str = f"{count} missed + {len(col_extra[j])} no data"
            else:
                count_str = f"{count}"
            ax.text(col_x[j] + col_w/2, 8.85, f"{col_headers[j]} ({count_str})",
                    fontsize=8, fontweight="bold", color="white", ha="center", va="center")

            # Column body background
            ax.add_patch(plt.Rectangle((col_x[j], 0.1), col_w, 8.4, fc=col_bg[j], ec=col_colors[j],
                                       linewidth=0.5, alpha=0.5))

            # List names
            names = [shorten(n, 28) for n in col_data[j]]
            if j == 1 and col_extra[j]:
                names.append("—— no pre-launch data ——")
                names.extend([shorten(n, 28) for n in col_extra[j]])

            # Truncate if too many for display
            max_display = 38
            truncated = False
            if len(names) > max_display:
                display_names = names[:max_display]
                truncated = True
            else:
                display_names = names

            y_start = 8.35
            line_h = 0.21
            for k, name in enumerate(display_names):
                y_pos = y_start - k * line_h
                if y_pos < 0.2:
                    break
                ax.text(col_x[j] + 0.08, y_pos, name,
                        fontsize=6, color=col_colors[j], va="top",
                        family="sans-serif", alpha=0.9)

            if truncated:
                remaining = len(names) - max_display
                ax.text(col_x[j] + 0.08, 0.3, f"... +{remaining} more",
                        fontsize=6, color="gray", va="top", style="italic")

    fig.suptitle("Peak Week Follower Velocity: 4-Tier Screening Strategy\n"
                 "Forward-observable | True Steam follower data | 25 evaluable + 5 no-data success games | 258 controls",
                 fontsize=14, fontweight="bold", y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "09_strategy_table.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("Saved output/09_strategy_table.png")


if __name__ == "__main__":
    run()
