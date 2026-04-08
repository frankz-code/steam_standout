"""Phase 7: Forward-looking threshold screening analysis.

Uses only metrics observable in real-time (before launch):
- Peak week velocity: best 7-day rolling avg of follower growth (ever seen so far)
- Average daily velocity: median 7-day rolling avg across all observations

Sweeps thresholds, computes precision/recall/F1, identifies optimal operating points.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FORWARD_FILE = DATA_DIR / "forward_observations.parquet"
CONFIG_DIR = PROJECT_ROOT / "config"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"


def load_data():
    fwd = pd.read_parquet(FORWARD_FILE)
    success_names = {str(g["steamId"]): g["name"]
                     for g in json.loads(SUCCESS_FILE.read_text())}
    bd = pd.read_parquet(BACKDROP_FILE)
    bd["steamId"] = bd["steamId"].astype(str)
    name_lookup = dict(zip(bd["steamId"], bd["name"]))
    name_lookup.update(success_names)
    return fwd, success_names, name_lookup


def build_per_game_forward_features(fwd: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game forward-observable features from time-series data.

    Only uses pre-launch follower observations (days_to_launch > 0).
    These are metrics you could compute at any point while watching a game.
    """
    fl = fwd[(fwd["metric"] == "followers") & (fwd["days_to_launch"] > 0)].copy()

    games = []
    for sid, g in fl.groupby("steamId"):
        group = g["group"].iloc[0]
        has_onset = g["onset_artifact"].iloc[0] if "onset_artifact" in g.columns else False

        rates = g["daily_growth_7d"].dropna()
        if has_onset and len(rates) > 1:
            rates = rates.iloc[1:]  # skip first rate (onset artifact)
        if rates.empty:
            continue

        games.append({
            "steamId": sid,
            "group": group,
            "peak_week_velocity": rates.max(),
            "avg_daily_velocity": rates.median(),
            "p75_daily_velocity": rates.quantile(0.75),
            "total_observations": len(rates),
            "max_followers": g["total"].max(),
        })

    return pd.DataFrame(games)


def sweep_threshold(gf: pd.DataFrame, col: str):
    """Sweep thresholds for a single feature."""
    n_success_total = (gf["group"] == "success").sum()
    if n_success_total == 0:
        return pd.DataFrame()

    values = gf[col].dropna().values
    # Percentile-based + absolute round numbers
    pct_thresholds = list(np.percentile(values, np.arange(5, 100, 2.5)))
    abs_thresholds = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150,
                      175, 200, 250, 300, 400, 500, 600, 750, 1000,
                      1250, 1500, 2000, 3000, 5000]
    thresholds = sorted(set(pct_thresholds + abs_thresholds))

    rows = []
    for t in thresholds:
        flagged = gf[gf[col] >= t]
        n_flagged = len(flagged)
        if n_flagged == 0:
            continue

        tp = (flagged["group"] == "success").sum()
        fp = (flagged["group"] == "control").sum()
        precision = tp / n_flagged
        recall = tp / n_success_total
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        rows.append({
            "threshold": t,
            "n_flagged": n_flagged,
            "true_positives": tp,
            "false_positives": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return pd.DataFrame(rows)


def plot_screening(all_results: dict):
    """Precision/recall curves for forward features."""
    features = [
        ("peak_week_velocity", "Peak week velocity (best 7d avg ever)"),
        ("avg_daily_velocity", "Avg daily velocity (median 7d avg)"),
        ("p75_daily_velocity", "P75 daily velocity (75th pctile of 7d avg)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {"peak_week_velocity": "#e74c3c",
              "avg_daily_velocity": "#3498db",
              "p75_daily_velocity": "#9b59b6"}

    # Top left: Precision vs Threshold
    ax = axes[0, 0]
    for col, label in features:
        if col not in all_results:
            continue
        r = all_results[col]
        ax.plot(r["threshold"], r["precision"], "-", label=label,
                linewidth=2, color=colors[col])
    ax.set_xlabel("Velocity threshold (followers/day)")
    ax.set_ylabel("Precision (hit rate)")
    ax.set_title("Of games we flag, what % are actual hits?", fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.3, label="50% hit rate")
    ax.axhline(0.33, color="gray", ls=":", alpha=0.3)

    # Top right: Recall vs Threshold
    ax = axes[0, 1]
    for col, label in features:
        if col not in all_results:
            continue
        r = all_results[col]
        ax.plot(r["threshold"], r["recall"], "-", label=label,
                linewidth=2, color=colors[col])
    ax.set_xlabel("Velocity threshold (followers/day)")
    ax.set_ylabel("Recall (catch rate)")
    ax.set_title("Of all viral hits, what % did we catch?", fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Bottom left: PR curve
    ax = axes[1, 0]
    for col, label in features:
        if col not in all_results:
            continue
        r = all_results[col]
        ax.plot(r["recall"], r["precision"], "o-", label=label,
                linewidth=2, markersize=2, color=colors[col])
    ax.set_xlabel("Recall (catch rate)")
    ax.set_ylabel("Precision (hit rate)")
    ax.set_title("Precision-Recall Tradeoff", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.1)

    # Bottom right: Radar size
    ax = axes[1, 1]
    for col, label in features:
        if col not in all_results:
            continue
        r = all_results[col]
        ax.plot(r["threshold"], r["n_flagged"], "-", label=label,
                linewidth=2, color=colors[col])
    ax.set_xlabel("Velocity threshold (followers/day)")
    ax.set_ylabel("Games on radar")
    ax.set_title("How many games to track?", fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    fig.suptitle("Forward-Looking Threshold Screening\n"
                 "Using only real-time observable metrics (follower growth, true Steam data)",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_threshold_screening.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 07_threshold_screening.png")


def plot_detail(gf: pd.DataFrame, all_results: dict, name_lookup: dict):
    """Detailed view of the best feature at key operating points."""
    for col, title_label in [
        ("peak_week_velocity", "Peak Week Velocity"),
        ("avg_daily_velocity", "Avg Daily Velocity"),
    ]:
        if col not in all_results:
            continue
        r = all_results[col]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: F1/precision/recall curves
        ax = axes[0]
        ax.plot(r["threshold"], r["precision"], "g-", label="Precision (hit rate)", linewidth=2)
        ax.plot(r["threshold"], r["recall"], "b-", label="Recall (catch rate)", linewidth=2)
        ax.plot(r["threshold"], r["f1"], "r--", label="F1 score", linewidth=2)
        ax.set_xlabel(f"{title_label} threshold (followers/day)")
        ax.set_ylabel("Score")
        ax.set_title(f"{title_label}: Operating Points", fontweight="bold")
        ax.set_xscale("log")
        ax.legend()
        ax.set_ylim(0, 1.05)

        # Mark best F1
        best_idx = r["f1"].idxmax()
        best = r.loc[best_idx]
        ax.axvline(best["threshold"], color="red", ls=":", alpha=0.5)
        ax.annotate(f"Best F1: {best['threshold']:.0f}/day\n"
                    f"P={best['precision']:.0%} R={best['recall']:.0%}\n"
                    f"Flagged: {best['n_flagged']:.0f}",
                    xy=(best["threshold"], best["f1"]),
                    xytext=(30, -30), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="red"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow"))

        # Mark 33% precision point
        p33 = r[r["precision"] >= 0.33]
        if not p33.empty:
            p33_row = p33.iloc[0]
            ax.axvline(p33_row["threshold"], color="green", ls=":", alpha=0.5)
            ax.annotate(f"1-in-3 hit rate: {p33_row['threshold']:.0f}/day\n"
                        f"R={p33_row['recall']:.0%}, n={p33_row['n_flagged']:.0f}",
                        xy=(p33_row["threshold"], 0.33),
                        xytext=(30, 30), textcoords="offset points", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="green"),
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen"))

        # Right: flagged games at best F1 threshold
        ax2 = axes[1]
        threshold = best["threshold"]
        flagged = gf[gf[col] >= threshold].sort_values(col, ascending=False).head(35)

        labels = []
        colors = []
        for _, row in flagged.iterrows():
            name = name_lookup.get(row["steamId"], row["steamId"])
            prefix = "+" if row["group"] == "success" else "-"
            labels.append(f"{prefix} {str(name)[:28]}")
            colors.append("#2ecc71" if row["group"] == "success" else "#e74c3c")

        ax2.barh(range(len(flagged)), flagged[col].values, color=colors, edgecolor="white")
        ax2.set_yticks(range(len(flagged)))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel(f"{title_label} (followers/day)")

        tp = (flagged["group"] == "success").sum()
        fp = (flagged["group"] == "control").sum()
        ax2.set_title(f"Games flagged at >={threshold:.0f}/day\n"
                      f"(green=hit, red=miss | {tp} hits / {tp+fp} flagged = "
                      f"{tp/(tp+fp):.0%} precision)", fontweight="bold")

        fig.suptitle(f"Screening Detail: {title_label} (forward-observable, true data)",
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        suffix = col.replace("_velocity", "").replace("_daily", "")
        fig.savefig(OUTPUT_DIR / f"08_threshold_detail_{suffix}.png", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved 08_threshold_detail_{suffix}.png")


def print_results(gf: pd.DataFrame, all_results: dict, success_names: dict, name_lookup: dict):
    """Print comprehensive results."""
    features = [
        ("peak_week_velocity", "Peak week velocity (best 7d avg ever seen)"),
        ("avg_daily_velocity", "Avg daily velocity (median of 7d avgs)"),
        ("p75_daily_velocity", "P75 daily velocity (75th pctile of 7d avgs)"),
    ]

    n_success = (gf["group"] == "success").sum()
    n_control = (gf["group"] == "control").sum()

    print(f"\n{'='*90}")
    print("FORWARD-LOOKING THRESHOLD SCREENING")
    print(f"{'='*90}")
    print(f"Games with forward follower data: {len(gf)} ({n_success} success, {n_control} control)")
    print(f"Metrics are real-time observable: computed from follower growth at any pre-launch point.")

    for col, label in features:
        if col not in all_results:
            continue
        r = all_results[col]

        s_vals = gf[gf["group"] == "success"][col]
        c_vals = gf[gf["group"] == "control"][col]

        print(f"\n{'─'*90}")
        print(f"  {label}")
        print(f"  Success median: {s_vals.median():.0f}/day | Control median: {c_vals.median():.0f}/day "
              f"| Ratio: {s_vals.median()/c_vals.median():.1f}x")
        print(f"{'─'*90}")
        print(f"  {'Threshold':>12} {'Flagged':>8} {'Hits':>5} {'FP':>5} "
              f"{'Precision':>10} {'Recall':>8} {'F1':>6}")
        print(f"  {'-'*62}")

        points = []

        # Best F1
        best_f1_idx = r["f1"].idxmax()
        points.append(("** Best F1 **", r.loc[best_f1_idx]))

        # Precision milestones
        for p_target in [0.25, 0.33, 0.50]:
            candidates = r[r["precision"] >= p_target]
            if not candidates.empty:
                row = candidates.iloc[0]
                points.append((f"P>={p_target:.0%}", row))

        # Recall milestones
        for r_target in [0.80, 0.60, 0.40]:
            candidates = r[r["recall"] >= r_target]
            if not candidates.empty:
                row = candidates.iloc[-1]
                points.append((f"R>={r_target:.0%}", row))

        seen = set()
        for point_label, row in points:
            key = f"{row['threshold']:.0f}"
            if key in seen:
                continue
            seen.add(key)
            print(f"  {row['threshold']:>10.0f}/d {row['n_flagged']:>8.0f} "
                  f"{row['true_positives']:>5.0f} {row['false_positives']:>5.0f} "
                  f"{row['precision']:>10.1%} {row['recall']:>8.1%} {row['f1']:>6.2f}"
                  f"  <- {point_label}")

    # Detailed breakdown for best feature
    # Find which feature has best F1
    best_feature = max(all_results.keys(),
                       key=lambda k: all_results[k]["f1"].max())
    best_r = all_results[best_feature]
    best_f1_idx = best_r["f1"].idxmax()
    threshold = best_r.loc[best_f1_idx, "threshold"]

    print(f"\n{'='*90}")
    print(f"DETAILED BREAKDOWN: {best_feature} >= {threshold:.0f}/day")
    print(f"{'='*90}")

    flagged = gf[gf[best_feature] >= threshold].sort_values(best_feature, ascending=False)

    print(f"\n  TRUE POSITIVES ({(flagged['group']=='success').sum()} success games caught):")
    tp = flagged[flagged["group"] == "success"]
    for _, row in tp.iterrows():
        name = name_lookup.get(row["steamId"], row["steamId"])
        print(f"    {row['steamId']:>10}  {str(name)[:38]:<39} peak={row['peak_week_velocity']:>7.0f}/d  "
              f"avg={row['avg_daily_velocity']:>6.0f}/d")

    print(f"\n  FALSE NEGATIVES ({n_success - len(tp)} success games MISSED):")
    missed = gf[(gf["group"] == "success") & (gf[best_feature] < threshold)]
    for _, row in missed.iterrows():
        name = name_lookup.get(row["steamId"], row["steamId"])
        print(f"    {row['steamId']:>10}  {str(name)[:38]:<39} peak={row['peak_week_velocity']:>7.0f}/d  "
              f"avg={row['avg_daily_velocity']:>6.0f}/d")

    # Success games not in forward data at all
    all_success_ids = set(success_names.keys())
    in_fwd = set(gf[gf["group"] == "success"]["steamId"])
    excluded = all_success_ids - in_fwd
    if excluded:
        print(f"\n  NOT EVALUABLE ({len(excluded)} success games with no pre-launch follower data):")
        for sid in excluded:
            print(f"    {sid:>10}  {success_names.get(sid, '?')}")

    print(f"\n  FALSE POSITIVES ({(flagged['group']=='control').sum()} control games flagged):")
    fp = flagged[flagged["group"] == "control"]
    for _, row in fp.iterrows():
        name = name_lookup.get(row["steamId"], row["steamId"])
        print(f"    {row['steamId']:>10}  {str(name)[:38]:<39} peak={row['peak_week_velocity']:>7.0f}/d  "
              f"avg={row['avg_daily_velocity']:>6.0f}/d")


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fwd, success_names, name_lookup = load_data()

    logger.info("Building per-game forward features...")
    gf = build_per_game_forward_features(fwd)
    logger.info(f"  {len(gf)} games with forward follower data "
                f"({(gf['group']=='success').sum()} success, {(gf['group']=='control').sum()} control)")

    features = ["peak_week_velocity", "avg_daily_velocity", "p75_daily_velocity"]
    all_results = {}
    for col in features:
        r = sweep_threshold(gf, col)
        if not r.empty:
            all_results[col] = r

    plot_screening(all_results)
    plot_detail(gf, all_results, name_lookup)
    print_results(gf, all_results, success_names, name_lookup)


if __name__ == "__main__":
    run()
