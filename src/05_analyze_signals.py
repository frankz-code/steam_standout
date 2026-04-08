"""Phase 5: Compare success vs control groups, generate charts and signal analysis."""

import json
import logging
import sys
from collections import Counter
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
ANALYSIS_FILE = DATA_DIR / "analysis_features.parquet"
TIMESERIES_FILE = DATA_DIR / "timeseries.parquet"
CONFIG_DIR = PROJECT_ROOT / "config"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"

# Plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(ANALYSIS_FILE)
    ts = pd.read_parquet(TIMESERIES_FILE) if TIMESERIES_FILE.exists() else pd.DataFrame()
    return df, ts


# --- Chart 1: Distribution Comparisons ---
def plot_distributions(df: pd.DataFrame):
    """Side-by-side distributions of pre-launch features for success vs control."""
    features = [
        ("wishlists_at_launch", "Wishlists at Launch"),
        ("followers_at_launch", "Followers at Launch"),
        ("wishlist_velocity_90d", "Wishlist Velocity (90d, per day)"),
        ("follower_velocity_90d", "Follower Velocity (90d, per day)"),
        ("days_on_steam_pre_launch", "Days on Steam Before Launch"),
        ("price_at_launch", "Price at Launch ($)"),
        ("reviews_at_launch", "Reviews at Launch (EA games)"),
        ("prior_game_count", "Developer's Prior Game Count"),
    ]

    available = [(col, label) for col, label in features if col in df.columns]
    n = len(available)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, (col, label) in enumerate(available):
        ax = axes[i]
        for group, color in [("success", "#2ecc71"), ("control", "#95a5a6")]:
            data = df[df["group"] == group][col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=15, alpha=0.6, label=group.title(), color=color, edgecolor="white")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pre-Launch Signal Distributions: Success vs Control", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_distributions.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 01_distributions.png")


# --- Chart 2: Time-Series Overlay ---
def plot_timeseries_overlay(df: pd.DataFrame, ts: pd.DataFrame):
    """Wishlist and follower trajectories aligned to launch date."""
    if ts.empty:
        logger.warning("No timeseries data, skipping overlay plot")
        return

    success_ids = set(df[df["group"] == "success"]["steamId"])
    control_ids = set(df[df["group"] == "control"]["steamId"])
    launch_dates = dict(zip(df["steamId"], pd.to_datetime(df["launch_date"])))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric, ax, title in [
        ("wishlists", axes[0], "Wishlist Trajectories (Aligned to Launch)"),
        ("followers", axes[1], "Follower Trajectories (Aligned to Launch)"),
    ]:
        for sid in success_ids | control_ids:
            game_ts = ts[ts["steamId"] == sid].sort_values("timestamp")
            if game_ts.empty or sid not in launch_dates:
                continue

            launch = launch_dates[sid]
            game_ts = game_ts.copy()
            game_ts["days_from_launch"] = (game_ts["timestamp"] - launch).dt.days

            # Only show pre-launch + first 30 days
            window = game_ts[(game_ts["days_from_launch"] >= -365) & (game_ts["days_from_launch"] <= 30)]
            if window.empty or metric not in window.columns:
                continue

            is_success = sid in success_ids
            color = "#2ecc71" if is_success else "#bdc3c7"
            alpha = 0.9 if is_success else 0.3
            lw = 2 if is_success else 0.8
            label_name = df[df["steamId"] == sid]["name"].iloc[0] if is_success else None

            ax.plot(window["days_from_launch"], window[metric],
                    color=color, alpha=alpha, linewidth=lw, label=label_name)

        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Launch Day")
        ax.set_xlabel("Days from Launch")
        ax.set_ylabel(metric.title())
        ax.set_title(title, fontweight="bold")
        if any(sid in success_ids for sid in launch_dates):
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Pre-Launch Trajectories: Success Games vs Control", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_timeseries_overlay.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 02_timeseries_overlay.png")


# --- Chart 3: Feature Separation Power ---
def plot_separation_analysis(df: pd.DataFrame):
    """Rank features by how well they separate success from control."""
    numeric_cols = [
        "wishlists_at_launch", "followers_at_launch",
        "wishlist_velocity_30d", "wishlist_velocity_90d", "wishlist_velocity_180d",
        "follower_velocity_30d", "follower_velocity_90d", "follower_velocity_180d",
        "days_on_steam_pre_launch", "price_at_launch", "reviews_at_launch",
        "tag_count", "prior_game_count", "prior_avg_revenue",
    ]
    available_cols = [c for c in numeric_cols if c in df.columns]

    separation_scores = []
    for col in available_cols:
        success_vals = df[df["group"] == "success"][col].dropna()
        control_vals = df[df["group"] == "control"][col].dropna()

        if len(success_vals) < 2 or len(control_vals) < 2:
            continue

        # Cohen's d effect size
        pooled_std = ((success_vals.std() ** 2 + control_vals.std() ** 2) / 2) ** 0.5
        if pooled_std > 0:
            cohens_d = abs(success_vals.mean() - control_vals.mean()) / pooled_std
        else:
            cohens_d = 0

        separation_scores.append({
            "feature": col,
            "cohens_d": cohens_d,
            "success_mean": success_vals.mean(),
            "control_mean": control_vals.mean(),
            "n_success": len(success_vals),
            "n_control": len(control_vals),
            "ratio": success_vals.mean() / control_vals.mean() if control_vals.mean() != 0 else float("inf"),
        })

    if not separation_scores:
        logger.warning("Not enough data for separation analysis")
        return

    sep_df = pd.DataFrame(separation_scores).sort_values("cohens_d", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(sep_df) * 0.5)))
    colors = ["#2ecc71" if d > 0.8 else "#f39c12" if d > 0.5 else "#95a5a6" for d in sep_df["cohens_d"]]
    ax.barh(sep_df["feature"], sep_df["cohens_d"], color=colors, edgecolor="white")
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11)
    ax.set_title("Feature Separation Power: Success vs Control\n(green = large effect, orange = medium, gray = small)",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=0.8, color="green", linestyle="--", alpha=0.3, label="Large effect threshold")
    ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.3, label="Medium effect threshold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_separation_power.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 03_separation_power.png")

    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE SEPARATION POWER (Cohen's d)")
    print("=" * 70)
    for _, row in sep_df.sort_values("cohens_d", ascending=False).iterrows():
        strength = "LARGE" if row["cohens_d"] > 0.8 else "MEDIUM" if row["cohens_d"] > 0.5 else "small"
        print(f"  {row['feature']:<35} d={row['cohens_d']:.2f} [{strength}]"
              f"  (success={row['success_mean']:.0f} vs control={row['control_mean']:.0f})"
              f"  n={row['n_success']:.0f}s/{row['n_control']:.0f}c")


# --- Chart 4: Tag Frequency Comparison ---
def plot_tag_analysis(df: pd.DataFrame):
    """Compare tag frequencies between success and control games."""
    success_tags = Counter()
    control_tags = Counter()
    n_success = (df["group"] == "success").sum()
    n_control = (df["group"] == "control").sum()

    for _, row in df.iterrows():
        tags = row.get("tags", [])
        if tags is None or (hasattr(tags, '__len__') and len(tags) == 0):
            continue
        counter = success_tags if row["group"] == "success" else control_tags
        for tag in tags:
            counter[tag] += 1

    # Find tags with biggest frequency differences
    all_tags = set(success_tags.keys()) | set(control_tags.keys())
    tag_diffs = []
    for tag in all_tags:
        s_pct = (success_tags.get(tag, 0) / n_success * 100) if n_success > 0 else 0
        c_pct = (control_tags.get(tag, 0) / n_control * 100) if n_control > 0 else 0
        tag_diffs.append({"tag": tag, "success_pct": s_pct, "control_pct": c_pct, "diff": s_pct - c_pct})

    tag_df = pd.DataFrame(tag_diffs).sort_values("diff", ascending=False)

    # Show top 15 over-represented and top 15 under-represented
    top = pd.concat([tag_df.head(15), tag_df.tail(15)])

    fig, ax = plt.subplots(figsize=(10, max(8, len(top) * 0.35)))
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in top["diff"]]
    ax.barh(top["tag"], top["diff"], color=colors, edgecolor="white")
    ax.set_xlabel("Percentage Point Difference (Success - Control)")
    ax.set_title("Tags Over/Under-Represented in Success Games", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_tag_analysis.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 04_tag_analysis.png")


# --- Chart 5: Developer Track Record ---
def plot_developer_track_record(df: pd.DataFrame):
    """Does prior developer success predict future hits?"""
    if "prior_game_count" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prior game count
    for group, color in [("success", "#2ecc71"), ("control", "#95a5a6")]:
        data = df[df["group"] == group]["prior_game_count"].dropna()
        axes[0].hist(data, bins=10, alpha=0.6, label=group.title(), color=color, edgecolor="white")
    axes[0].set_title("Developer's Prior Game Count", fontweight="bold")
    axes[0].set_xlabel("Number of Prior Games")
    axes[0].legend()

    # Prior avg revenue
    if "prior_avg_revenue" in df.columns:
        for group, color in [("success", "#2ecc71"), ("control", "#95a5a6")]:
            data = df[df["group"] == group]["prior_avg_revenue"].dropna()
            data = data[data > 0]  # exclude zeros (no prior games)
            if not data.empty:
                axes[1].hist(data, bins=15, alpha=0.6, label=group.title(), color=color, edgecolor="white")
        axes[1].set_title("Developer's Prior Avg Revenue", fontweight="bold")
        axes[1].set_xlabel("Average Revenue ($)")
        axes[1].legend()

    fig.suptitle("Developer Track Record: Success vs Control", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_developer_track_record.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 05_developer_track_record.png")


# --- Chart 6: Scatter Matrix ---
def plot_scatter_matrix(df: pd.DataFrame):
    """Scatter plot of key features colored by group."""
    key_features = ["wishlists_at_launch", "followers_at_launch",
                    "wishlist_velocity_90d", "first_month_revenue"]
    available = [f for f in key_features if f in df.columns]

    if len(available) < 2:
        return

    plot_df = df[available + ["group"]].dropna()
    if plot_df.empty:
        return

    g = sns.pairplot(plot_df, hue="group",
                     palette={"success": "#2ecc71", "control": "#95a5a6"},
                     diag_kind="hist", plot_kws={"alpha": 0.7})
    g.fig.suptitle("Key Feature Relationships", y=1.02, fontsize=14, fontweight="bold")
    g.savefig(OUTPUT_DIR / "06_scatter_matrix.png", bbox_inches="tight")
    plt.close(g.fig)
    logger.info("Saved 06_scatter_matrix.png")


def print_summary(df: pd.DataFrame):
    """Print strongest signals found."""
    print("\n" + "=" * 70)
    print("SIGNAL ANALYSIS SUMMARY")
    print("=" * 70)

    success = df[df["group"] == "success"]
    control = df[df["group"] == "control"]

    print(f"\nDataset: {len(success)} success games, {len(control)} control games")

    key_metrics = [
        ("wishlists_at_launch", "Wishlists at Launch"),
        ("followers_at_launch", "Followers at Launch"),
        ("wishlist_velocity_90d", "Wishlist Velocity (90d)"),
        ("follower_velocity_90d", "Follower Velocity (90d)"),
        ("first_month_revenue", "First Month Revenue"),
        ("first_month_copies", "First Month Copies Sold"),
    ]

    print(f"\n{'Metric':<30} {'Success (median)':>18} {'Control (median)':>18} {'Ratio':>8}")
    print("-" * 78)
    for col, label in key_metrics:
        if col not in df.columns:
            continue
        s_med = success[col].dropna().median()
        c_med = control[col].dropna().median()
        ratio = s_med / c_med if c_med != 0 else float("inf")
        print(f"{label:<30} {s_med:>18,.0f} {c_med:>18,.0f} {ratio:>7.1f}x")

    print("\n" + "=" * 70)
    print("Charts saved to output/")
    print("=" * 70)


def analyze_signals():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ANALYSIS_FILE.exists():
        print("Error: Run 04_build_analysis_set.py first")
        sys.exit(1)

    df, ts = load_data()
    logger.info(f"Loaded {len(df)} games ({(df['group']=='success').sum()} success, "
                f"{(df['group']=='control').sum()} control)")

    plot_distributions(df)
    plot_timeseries_overlay(df, ts)
    plot_separation_analysis(df)
    plot_tag_analysis(df)
    plot_developer_track_record(df)
    plot_scatter_matrix(df)
    print_summary(df)


if __name__ == "__main__":
    analyze_signals()
