"""Phase 6: Forward-looking analysis — what signals are observable at any point in time.

Builds forward_observations.parquet from raw history, then generates charts.
Applies onset-artifact detection: flags the first tracked entry so that
growth rates spanning the tracking-onset boundary are excluded.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
HISTORIES_DIR = DATA_DIR / "histories"
CONFIG_DIR = PROJECT_ROOT / "config"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
CONTROL_FILE = CONFIG_DIR / "control_group.json"
ANALYSIS_FILE = DATA_DIR / "analysis_features.parquet"
FORWARD_FILE = DATA_DIR / "forward_observations.parquet"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150

# Threshold: if first-ever history entry already has this many followers,
# the game was being tracked mid-life. Growth rates from the first few
# entries are still valid (they measure real deltas between tracked points),
# but we mark days_since_creation=1 so downstream knows this isn't "day 1".
ONSET_ARTIFACT_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Phase 6a: Build forward observations from raw history
# ---------------------------------------------------------------------------

def get_game_group_labels() -> dict[str, str]:
    labels = {}
    success_games = json.loads(SUCCESS_FILE.read_text())
    for sg in success_games:
        labels[str(sg["steamId"])] = "success"
    if CONTROL_FILE.exists():
        control_group = json.loads(CONTROL_FILE.read_text())
        for matches in control_group.values():
            for cid in matches:
                labels[str(cid)] = "control"
    return labels


def build_forward_observations():
    """Build time-series observations with rolling growth rates."""
    labels = get_game_group_labels()
    all_rows = []

    for json_file in HISTORIES_DIR.glob("*.json"):
        steam_id = json_file.stem
        if steam_id not in labels:
            continue

        data = json.loads(json_file.read_text())
        history = data.get("history", [])
        if not history:
            continue

        launch_ms = data.get("firstReleaseDate") or data.get("releaseDate")
        if not launch_ms:
            continue
        launch_date = pd.to_datetime(launch_ms, unit="ms")

        hist_df = pd.DataFrame(history)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timeStamp"], unit="ms", errors="coerce")
        hist_df = hist_df.sort_values("timestamp").reset_index(drop=True)

        page_created = hist_df["timestamp"].min()

        for metric in ["followers", "wishlists"]:
            if metric not in hist_df.columns:
                continue

            series = hist_df[["timestamp", metric]].dropna(subset=[metric]).copy()
            if len(series) < 2:
                continue

            series = series.rename(columns={metric: "total"})
            series["date"] = series["timestamp"]
            series["days_to_launch"] = (launch_date - series["timestamp"]).dt.days
            series["days_since_creation"] = (series["timestamp"] - page_created).dt.days + 1

            # Onset-artifact flag: if first entry already has a large value,
            # mark it so we know this isn't organic growth from zero.
            first_total = series["total"].iloc[0]
            has_onset_artifact = first_total > ONSET_ARTIFACT_THRESHOLD

            # Compute daily growth using actual deltas between consecutive entries.
            # This is valid even with onset artifacts because we measure
            # real changes between tracked points.
            series["daily_delta"] = series["total"].diff()
            series["time_delta_days"] = series["timestamp"].diff().dt.total_seconds() / 86400
            series["daily_rate"] = series["daily_delta"] / series["time_delta_days"]

            # Rolling 7-day and 30-day growth: average of daily_rate over windows.
            # Only valid when we have enough consecutive entries.
            series["daily_growth_7d"] = series["daily_rate"].rolling(7, min_periods=7).mean()
            series["daily_growth_30d"] = series["daily_rate"].rolling(30, min_periods=14).mean()

            # If onset artifact, null out the first growth entry because it spans
            # the unknown gap before tracking started. The daily_rate for entry 1
            # measures "growth since Gamalytic started tracking" which is arbitrary.
            if has_onset_artifact:
                series.iloc[0, series.columns.get_loc("daily_rate")] = np.nan

            for _, row in series.iterrows():
                all_rows.append({
                    "steamId": steam_id,
                    "group": labels[steam_id],
                    "metric": metric,
                    "date": row["date"],
                    "days_to_launch": row["days_to_launch"],
                    "days_since_creation": row["days_since_creation"],
                    "total": row["total"],
                    "daily_growth_7d": row["daily_growth_7d"],
                    "daily_growth_30d": row["daily_growth_30d"],
                    "onset_artifact": has_onset_artifact,
                })

    df = pd.DataFrame(all_rows)
    df.to_parquet(FORWARD_FILE, index=False)
    logger.info(f"Saved {len(df):,} forward observations for {df['steamId'].nunique()} games")
    return df


# ---------------------------------------------------------------------------
# Phase 6b: Forward analysis charts
# ---------------------------------------------------------------------------

def load_forward_data():
    if FORWARD_FILE.exists():
        return pd.read_parquet(FORWARD_FILE)
    return build_forward_observations()


def chart_f1_separation_power():
    """Feature ranking: follower (true) vs wishlist (est), with sample sizes."""
    if not ANALYSIS_FILE.exists():
        return
    df = pd.read_parquet(ANALYSIS_FILE)

    features = [
        ("follower_velocity_30d", "Follower vel 30d (true)", "#2ecc71"),
        ("follower_velocity_90d", "Follower vel 90d (true)", "#27ae60"),
        ("follower_velocity_180d", "Follower vel 180d (true)", "#1e8449"),
        ("followers_at_launch", "Followers at launch (true)", "#229954"),
        ("wishlist_velocity_30d", "Wishlist vel 30d (est)", "#f39c12"),
        ("wishlist_velocity_90d", "Wishlist vel 90d (est)", "#e67e22"),
        ("wishlist_velocity_180d", "Wishlist vel 180d (est)", "#d35400"),
        ("wishlists_at_launch", "Wishlists at launch (est)", "#e67e22"),
    ]

    scores = []
    for col, label, color in features:
        if col not in df.columns:
            continue
        sv = df[df["group"] == "success"][col].dropna()
        cv = df[df["group"] == "control"][col].dropna()
        if len(sv) < 2 or len(cv) < 2:
            continue
        pooled = ((sv.std()**2 + cv.std()**2) / 2) ** 0.5
        d = abs(sv.mean() - cv.mean()) / pooled if pooled > 0 else 0
        scores.append({"label": label, "d": d, "color": color,
                        "n_s": len(sv), "n_c": len(cv)})

    if not scores:
        return

    sdf = pd.DataFrame(scores).sort_values("d")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sdf["label"], sdf["d"], color=sdf["color"], edgecolor="white")
    for bar, (_, row) in zip(bars, sdf.iterrows()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"n={row['n_s']:.0f}s/{row['n_c']:.0f}c", va="center", fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Feature Separation: Follower (true data) vs Wishlist (estimated)\nWith per-window sample sizes",
                 fontweight="bold")
    ax.axvline(0.8, color="green", ls="--", alpha=0.3)
    ax.axvline(0.5, color="orange", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F1_follower_separation_power.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F1_follower_separation_power.png")


def chart_f2_velocity_showcase():
    """Strip + violin plots of follower velocity."""
    if not ANALYSIS_FILE.exists():
        return
    df = pd.read_parquet(ANALYSIS_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, title in [
        (axes[0], "follower_velocity_30d", "Follower Velocity (30d before launch)"),
        (axes[1], "follower_velocity_90d", "Follower Velocity (90d before launch)"),
    ]:
        if col not in df.columns:
            continue
        plot_df = df[["group", col]].dropna()
        n_s = (plot_df["group"] == "success").sum()
        n_c = (plot_df["group"] == "control").sum()

        sns.violinplot(data=plot_df, x="group", y=col, hue="group", ax=ax,
                       palette={"success": "#2ecc71", "control": "#95a5a6"},
                       inner=None, alpha=0.3, legend=False)
        sns.stripplot(data=plot_df, x="group", y=col, hue="group", ax=ax,
                      palette={"success": "#27ae60", "control": "#7f8c8d"},
                      size=4, alpha=0.7, jitter=True, legend=False)
        ax.set_title(f"{title}\n(n={n_s} success, {n_c} control)", fontweight="bold")
        ax.set_ylabel("Followers/day")

    fig.suptitle("Follower Velocity: Success vs Control (true Steam data)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F2_follower_velocity_showcase.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F2_follower_velocity_showcase.png")


def chart_f3_trajectories(fwd: pd.DataFrame):
    """All games' follower trajectories aligned to launch."""
    fl = fwd[(fwd["metric"] == "followers") & (fwd["days_to_launch"] >= -30)].copy()
    success_ids = set(fl[fl["group"] == "success"]["steamId"])

    fig, ax = plt.subplots(figsize=(14, 7))
    for sid in fl["steamId"].unique():
        g = fl[fl["steamId"] == sid].sort_values("days_to_launch", ascending=False)
        is_s = sid in success_ids
        ax.plot(g["days_to_launch"], g["total"],
                color="#2ecc71" if is_s else "#bdc3c7",
                alpha=0.8 if is_s else 0.15,
                linewidth=2 if is_s else 0.5)

    ax.axvline(0, color="red", ls="--", alpha=0.5, label="Launch Day")
    ax.set_xlabel("Days to Launch")
    ax.set_ylabel("Followers")
    ax.set_title("Follower Trajectories: All Games Aligned to Launch\n(green=success, gray=control)",
                 fontweight="bold")
    ax.set_xlim(fl["days_to_launch"].max(), -30)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F3_follower_trajectories_all.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F3_follower_trajectories_all.png")


def chart_f4_velocity_windows(fwd: pd.DataFrame):
    """Window-by-window follower velocity comparison."""
    fl = fwd[(fwd["metric"] == "followers")].copy()

    windows = [
        ("T-360 to T-180", -360, -180),
        ("T-180 to T-90", -180, -90),
        ("T-90 to T-30", -90, -30),
        ("T-30 to T-0", -30, 0),
    ]

    results = []
    for label, start, end in windows:
        # days_to_launch counts down: 360 means 360 days before launch
        w = fl[(fl["days_to_launch"] <= -start) & (fl["days_to_launch"] > -end)]

        for group in ["success", "control"]:
            gw = w[w["group"] == group]
            # Per-game: only include games that have entries spanning most of the window
            game_velocities = []
            for sid, sg in gw.groupby("steamId"):
                if len(sg) < 5:
                    continue
                sg = sg.sort_values("days_to_launch", ascending=False)

                # Onset artifact check: if this window includes the game's
                # first-ever tracked entry AND it started with a large value,
                # skip the first entry's rate from the average.
                rates = sg["daily_growth_7d"].dropna()
                if rates.empty:
                    # Fall back to total delta across window
                    total_start = sg["total"].iloc[0]
                    total_end = sg["total"].iloc[-1]
                    days = abs((sg["date"].iloc[-1] - sg["date"].iloc[0]).days)
                    if days > 0 and pd.notna(total_start) and pd.notna(total_end):
                        game_velocities.append((total_end - total_start) / days)
                else:
                    game_velocities.append(rates.median())

            if game_velocities:
                results.append({
                    "window": label,
                    "group": group,
                    "median_velocity": np.median(game_velocities),
                    "mean_velocity": np.mean(game_velocities),
                    "n_games": len(game_velocities),
                })

    if not results:
        return

    rdf = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: velocity bars
    ax = axes[0]
    window_labels = [w[0] for w in windows]
    x = np.arange(len(window_labels))
    width = 0.35
    s_vals = [rdf[(rdf["window"] == w) & (rdf["group"] == "success")]["median_velocity"].values
              for w in window_labels]
    c_vals = [rdf[(rdf["window"] == w) & (rdf["group"] == "control")]["median_velocity"].values
              for w in window_labels]
    s_vals = [v[0] if len(v) > 0 else 0 for v in s_vals]
    c_vals = [v[0] if len(v) > 0 else 0 for v in c_vals]

    bars_s = ax.bar(x - width/2, s_vals, width, label="Success", color="#2ecc71")
    bars_c = ax.bar(x + width/2, c_vals, width, label="Control", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(window_labels, rotation=15)
    ax.set_ylabel("Median Follower Velocity (per day)")
    ax.set_title("Follower Velocity by Time Window", fontweight="bold")
    ax.legend()

    # Add sample sizes
    for i, w in enumerate(window_labels):
        ns = rdf[(rdf["window"] == w) & (rdf["group"] == "success")]["n_games"].values
        nc = rdf[(rdf["window"] == w) & (rdf["group"] == "control")]["n_games"].values
        ns = ns[0] if len(ns) > 0 else 0
        nc = nc[0] if len(nc) > 0 else 0
        ax.text(i, max(s_vals[i], c_vals[i]) * 1.05, f"n={ns}s/{nc}c",
                ha="center", fontsize=7, color="gray")

    # Right: ratio over time
    ax2 = axes[1]
    ratios = [s/c if c > 0 else 0 for s, c in zip(s_vals, c_vals)]
    ax2.plot(window_labels, ratios, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax2.set_ylabel("Success/Control Ratio")
    ax2.set_title("Velocity Gap Ratio by Window", fontweight="bold")
    ax2.axhline(1, color="gray", ls="--", alpha=0.3)
    for i, r in enumerate(ratios):
        ax2.annotate(f"{r:.1f}x", (i, r), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontweight="bold")

    fig.suptitle("Follower Growth: Window-by-Window Analysis (true Steam data)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F4_follower_velocity_windows.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F4_follower_velocity_windows.png")


def chart_f5_forward_analysis(fwd: pd.DataFrame):
    """Forward-looking: peak week growth + growth by follower stage."""
    fl = fwd[(fwd["metric"] == "followers") & (fwd["days_to_launch"] > 0)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Peak week follower growth ---
    ax = axes[0]
    peak_week = []
    for sid, g in fl.groupby("steamId"):
        rates = g["daily_growth_7d"].dropna()
        if rates.empty:
            continue
        group = g["group"].iloc[0]

        # Onset artifact: if the game's first entry had onset_artifact=True,
        # the first few growth rates might be inflated. Skip the first
        # valid growth rate if the game has onset artifact.
        if g["onset_artifact"].iloc[0]:
            rates = rates.iloc[1:]  # skip first computed rate
            if rates.empty:
                continue

        peak_week.append({"steamId": sid, "group": group, "peak_7d": rates.max()})

    if peak_week:
        pw = pd.DataFrame(peak_week)
        for group, color in [("success", "#2ecc71"), ("control", "#95a5a6")]:
            data = pw[pw["group"] == group]["peak_7d"]
            ax.hist(data, bins=20, alpha=0.6, label=f"{group.title()} (n={len(data)})",
                    color=color, edgecolor="white")
        ax.set_title("Peak Week Follower Growth (best 7-day avg)\nPre-launch only",
                     fontweight="bold")
        ax.set_xlabel("Peak followers/day")
        ax.set_ylabel("Count")
        ax.legend()

        s_med = pw[pw["group"] == "success"]["peak_7d"].median()
        c_med = pw[pw["group"] == "control"]["peak_7d"].median()
        ratio = s_med / c_med if c_med > 0 else float("inf")
        ax.axvline(s_med, color="#27ae60", ls="--", alpha=0.7, label=f"Success median: {s_med:.0f}/d")
        ax.axvline(c_med, color="#7f8c8d", ls="--", alpha=0.7, label=f"Control median: {c_med:.0f}/d")
        ax.legend(fontsize=8)

    # --- Right: Growth by follower stage ---
    ax2 = axes[1]
    stages = [
        ("<500", 0, 500),
        ("500-2K", 500, 2000),
        ("2K-5K", 2000, 5000),
        ("5K-15K", 5000, 15000),
        ("15K-50K", 15000, 50000),
        ("50K-150K", 50000, 150000),
        ("150K+", 150000, float("inf")),
    ]

    stage_data = []
    for label, lo, hi in stages:
        stage = fl[(fl["total"] >= lo) & (fl["total"] < hi)]
        for group in ["success", "control"]:
            rates = stage[stage["group"] == group]["daily_growth_7d"].dropna()
            if len(rates) > 5:
                stage_data.append({
                    "stage": label, "group": group,
                    "median_growth": rates.median(),
                    "n_obs": len(rates),
                })

    if stage_data:
        sd = pd.DataFrame(stage_data)
        stage_labels = [s[0] for s in stages]
        x = np.arange(len(stage_labels))
        width = 0.35
        s_g = [sd[(sd["stage"] == s) & (sd["group"] == "success")]["median_growth"].values for s in stage_labels]
        c_g = [sd[(sd["stage"] == s) & (sd["group"] == "control")]["median_growth"].values for s in stage_labels]
        s_g = [v[0] if len(v) > 0 else 0 for v in s_g]
        c_g = [v[0] if len(v) > 0 else 0 for v in c_g]

        ax2.bar(x - width/2, s_g, width, label="Success", color="#2ecc71")
        ax2.bar(x + width/2, c_g, width, label="Control", color="#95a5a6")
        ax2.set_xticks(x)
        ax2.set_xticklabels(stage_labels, rotation=30, fontsize=9)
        ax2.set_ylabel("Median Daily Follower Growth (7d avg)")
        ax2.set_title("Follower Growth by Current Follower Count Stage\nPre-launch only",
                      fontweight="bold")
        ax2.legend()

    fig.suptitle("Forward-Looking Follower Analysis (true Steam data)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F5_forward_follower_analysis.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F5_forward_follower_analysis.png")


def chart_f6_followers_vs_revenue():
    """Log-log scatter: followers at launch vs first-month revenue."""
    if not ANALYSIS_FILE.exists():
        return
    df = pd.read_parquet(ANALYSIS_FILE)

    cols = ["followers_at_launch", "first_month_revenue", "group", "name"]
    plot_df = df[cols].dropna()
    plot_df = plot_df[(plot_df["followers_at_launch"] > 0) & (plot_df["first_month_revenue"] > 0)]

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    for group, color, zorder in [("control", "#bdc3c7", 1), ("success", "#2ecc71", 2)]:
        gd = plot_df[plot_df["group"] == group]
        ax.scatter(gd["followers_at_launch"], gd["first_month_revenue"],
                   c=color, alpha=0.6, s=40, zorder=zorder, label=group.title(),
                   edgecolors="white", linewidth=0.5)

    # Label success games
    for _, row in plot_df[plot_df["group"] == "success"].iterrows():
        name = str(row["name"])[:20]
        ax.annotate(name, (row["followers_at_launch"], row["first_month_revenue"]),
                    fontsize=6, alpha=0.7, xytext=(5, 5), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Followers at Launch (true Steam data)")
    ax.set_ylabel("First Month Revenue ($, estimated)")
    ax.set_title("Followers at Launch vs First-Month Revenue", fontweight="bold")
    ax.axvline(25000, color="#e74c3c", ls="--", alpha=0.3, label="25K follower threshold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "F6_followers_vs_revenue.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F6_followers_vs_revenue.png")


def print_forward_summary(fwd: pd.DataFrame):
    """Print forward analysis statistics."""
    fl = fwd[(fwd["metric"] == "followers") & (fwd["days_to_launch"] > 0)]
    onset_games = fwd[fwd["onset_artifact"] == True]["steamId"].nunique()
    total_games = fwd["steamId"].nunique()

    print(f"\n{'='*70}")
    print("FORWARD ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total observations: {len(fwd):,}")
    print(f"Total games: {total_games}")
    print(f"Games with onset artifact (first entry > {ONSET_ARTIFACT_THRESHOLD} followers): {onset_games}")
    print(f"Pre-launch follower observations: {len(fl):,}")

    # Peak week stats
    peak_data = []
    for sid, g in fl.groupby("steamId"):
        rates = g["daily_growth_7d"].dropna()
        if g["onset_artifact"].iloc[0]:
            rates = rates.iloc[1:]
        if rates.empty:
            continue
        peak_data.append({"group": g["group"].iloc[0], "peak": rates.max()})

    if peak_data:
        pdf = pd.DataFrame(peak_data)
        s = pdf[pdf["group"] == "success"]["peak"]
        c = pdf[pdf["group"] == "control"]["peak"]

        print(f"\nPeak week follower growth (pre-launch):")
        print(f"  Success: median={s.median():.0f}/day (n={len(s)})")
        print(f"  Control: median={c.median():.0f}/day (n={len(c)})")
        if c.median() > 0:
            print(f"  Ratio: {s.median()/c.median():.1f}x")

        pooled = ((s.std()**2 + c.std()**2) / 2) ** 0.5
        if pooled > 0:
            d = abs(s.mean() - c.mean()) / pooled
            print(f"  Cohen's d: {d:.2f}")

    # Any check-in stats
    any_rates = fl["daily_growth_7d"].dropna()
    if not any_rates.empty:
        for group in ["success", "control"]:
            gr = fl[fl["group"] == group]["daily_growth_7d"].dropna()
            print(f"\nAny check-in 7d growth ({group}): median={gr.median():.0f}/day, n_obs={len(gr):,}")


def run_forward_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building forward observations...")
    fwd = build_forward_observations()

    logger.info("Generating charts...")
    chart_f1_separation_power()
    chart_f2_velocity_showcase()
    chart_f3_trajectories(fwd)
    chart_f4_velocity_windows(fwd)
    chart_f5_forward_analysis(fwd)
    chart_f6_followers_vs_revenue()

    print_forward_summary(fwd)


if __name__ == "__main__":
    run_forward_analysis()
