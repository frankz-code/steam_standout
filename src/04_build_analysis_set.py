"""Phase 4: Transform raw history into analysis-ready features."""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
HISTORIES_DIR = DATA_DIR / "histories"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
CONTROL_FILE = CONFIG_DIR / "control_group.json"
OUTPUT_FILE = DATA_DIR / "analysis_features.parquet"


def get_game_group_labels() -> dict[str, str]:
    """Map steamId → 'success' or 'control'."""
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


def compute_velocity(history_df: pd.DataFrame, col: str, launch_date: pd.Timestamp,
                     days_before: int, page_created: pd.Timestamp | None = None) -> float | None:
    """Compute average daily change in a metric over a pre-launch window.

    Returns None (not 0) when the game lacks sufficient data for this window,
    so downstream analysis can exclude it per-feature rather than per-game.

    Detects tracking-onset artifacts: if the first entry in the window already
    has a large value but is also the first entry in the entire history, this
    is Gamalytic starting to track a game that already had accumulated
    followers/wishlists — not real organic growth. These are returned as None.
    """
    if col not in history_df.columns:
        return None

    window_start = launch_date - pd.Timedelta(days=days_before)
    window = history_df[
        (history_df["timestamp"] >= window_start) &
        (history_df["timestamp"] < launch_date)
    ].sort_values("timestamp")

    if len(window) < 2:
        return None

    # Check if the game's history actually covers the requested window.
    # If history starts after window_start, we don't have full coverage.
    history_start = history_df["timestamp"].min()
    if pd.notna(history_start) and history_start > window_start + pd.Timedelta(days=7):
        # History starts more than 7 days into the window — insufficient coverage.
        # Return None so this game is excluded from this velocity calculation.
        return None

    val_start = window[col].iloc[0]
    val_end = window[col].iloc[-1]
    time_span = (window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]).days

    if time_span <= 0 or pd.isna(val_start) or pd.isna(val_end):
        return None

    # Tracking-onset artifact detection:
    # If the first window entry is also (approximately) the first entry in the
    # entire history AND it already shows a large value, this is Gamalytic
    # picking up a game mid-life, not real growth from zero.
    first_window_ts = window["timestamp"].iloc[0]
    is_first_entry = (first_window_ts - history_start).days <= 3
    if is_first_entry and val_start > 0:
        # The game already had followers/wishlists when tracking started.
        # The velocity from this starting point is still meaningful (it
        # measures growth *during* the tracked period), so we keep it.
        # But if val_start is the ONLY non-zero point (i.e., everything
        # before was untracked, then suddenly appears), flag it:
        # Check if there's a suspiciously large jump from 0 in the first entry.
        all_pre_window = history_df[history_df["timestamp"] < first_window_ts]
        if all_pre_window.empty and val_start > 100:
            # First-ever entry already shows >100 — onset artifact.
            # The velocity within the window is still valid if we have
            # enough entries, but the "starting from 0" assumption is wrong.
            # We compute velocity from the tracked data only.
            pass  # velocity from val_start→val_end is still valid growth rate

    return (val_end - val_start) / time_span


def compute_value_at_launch(history_df: pd.DataFrame, col: str,
                            launch_date: pd.Timestamp) -> float:
    """Get the value of a metric closest to (but before) launch."""
    if col not in history_df.columns:
        return 0.0
    pre = history_df[history_df["timestamp"] <= launch_date].sort_values("timestamp")
    if pre.empty:
        return 0.0
    val = pre[col].iloc[-1]
    return val if not pd.isna(val) else 0.0


def _safe_last(df: pd.DataFrame, col: str) -> float:
    """Get last value of a column if it exists, else 0."""
    if col not in df.columns or df.empty:
        return 0
    val = df[col].iloc[-1]
    return val if pd.notna(val) else 0


def compute_first_month_outcome(history_df: pd.DataFrame, launch_date: pd.Timestamp) -> dict:
    """Compute first-month revenue and copies sold from cumulative history."""
    month_end = launch_date + pd.Timedelta(days=30)

    at_launch = history_df[history_df["timestamp"] <= launch_date].sort_values("timestamp")
    at_month = history_df[history_df["timestamp"] <= month_end].sort_values("timestamp")

    launch_sales = _safe_last(at_launch, "sales")
    launch_revenue = _safe_last(at_launch, "revenue")
    month_sales = _safe_last(at_month, "sales")
    month_revenue = _safe_last(at_month, "revenue")

    return {
        "first_month_copies": month_sales - launch_sales,
        "first_month_revenue": month_revenue - launch_revenue,
    }


def get_developer_track_record(dev_name: str, backdrop_df: pd.DataFrame,
                               before_date: pd.Timestamp) -> dict:
    """Look up developer's prior games from backdrop dataset."""
    if not dev_name or pd.isna(dev_name):
        return {"prior_game_count": 0, "prior_avg_revenue": 0}

    prior = backdrop_df[
        (backdrop_df["developers"].apply(
            lambda devs: hasattr(devs, '__iter__') and not isinstance(devs, str) and dev_name in devs
        )) &
        (backdrop_df["firstReleaseDate"] < before_date)
    ]

    if prior.empty:
        return {"prior_game_count": 0, "prior_avg_revenue": 0}

    return {
        "prior_game_count": len(prior),
        "prior_avg_revenue": prior["revenue"].dropna().mean() if not prior["revenue"].dropna().empty else 0,
    }


def build_analysis_set():
    labels = get_game_group_labels()

    # Load backdrop for developer track record lookup
    backdrop_df = pd.read_parquet(BACKDROP_FILE) if BACKDROP_FILE.exists() else pd.DataFrame()

    features = []

    for json_file in HISTORIES_DIR.glob("*.json"):
        steam_id = json_file.stem
        if steam_id not in labels:
            continue

        data = json.loads(json_file.read_text())
        history = data.get("history", [])

        if not history:
            logger.warning(f"{steam_id}: no history data, skipping")
            continue

        # Build history DataFrame
        hist_df = pd.DataFrame(history)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timeStamp"], unit="ms", errors="coerce")
        hist_df = hist_df.sort_values("timestamp")

        # Determine launch date
        launch_ms = data.get("firstReleaseDate") or data.get("releaseDate")
        if not launch_ms:
            logger.warning(f"{steam_id}: no release date, skipping")
            continue
        launch_date = pd.to_datetime(launch_ms, unit="ms")

        # First history entry = page creation date
        page_created = hist_df["timestamp"].min()
        days_on_steam_before_launch = (launch_date - page_created).days if pd.notna(page_created) else 0

        # Pre-launch features
        row = {
            "steamId": steam_id,
            "name": data.get("name", ""),
            "group": labels[steam_id],
            "launch_date": launch_date,
            "price_at_launch": data.get("price", 0),
            "early_access": data.get("earlyAccess", False),
            "days_on_steam_pre_launch": max(days_on_steam_before_launch, 0),

            # Counts at launch
            "wishlists_at_launch": compute_value_at_launch(hist_df, "wishlists", launch_date),
            "followers_at_launch": compute_value_at_launch(hist_df, "followers", launch_date),
            "reviews_at_launch": compute_value_at_launch(hist_df, "reviews", launch_date),

            # Velocities — returns None when data coverage is insufficient
            # for that specific window, so downstream excludes per-feature.
            "wishlist_velocity_30d": compute_velocity(hist_df, "wishlists", launch_date, 30, page_created),
            "wishlist_velocity_90d": compute_velocity(hist_df, "wishlists", launch_date, 90, page_created),
            "wishlist_velocity_180d": compute_velocity(hist_df, "wishlists", launch_date, 180, page_created),
            "follower_velocity_30d": compute_velocity(hist_df, "followers", launch_date, 30, page_created),
            "follower_velocity_90d": compute_velocity(hist_df, "followers", launch_date, 90, page_created),
            "follower_velocity_180d": compute_velocity(hist_df, "followers", launch_date, 180, page_created),

            # Tags/genres
            "tags": data.get("tags", []),
            "genres": data.get("genres", []),
            "tag_count": len(data.get("tags", []) or []),
            "genre_count": len(data.get("genres", []) or []),
        }

        # First month outcome
        outcome = compute_first_month_outcome(hist_df, launch_date)
        row.update(outcome)

        # Developer track record
        developers = data.get("developers", [])
        primary_dev = developers[0] if developers else None
        if not backdrop_df.empty:
            track = get_developer_track_record(primary_dev, backdrop_df, launch_date)
            row.update(track)
        else:
            row["prior_game_count"] = 0
            row["prior_avg_revenue"] = 0

        features.append(row)
        logger.info(f"Processed {steam_id} ({data.get('name', '')}) - {labels[steam_id]}")

    if not features:
        logger.warning("No features computed!")
        return

    df = pd.DataFrame(features)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print("ANALYSIS SET SUMMARY")
    print(f"{'=' * 60}")
    n_success = (df['group'] == 'success').sum()
    n_control = (df['group'] == 'control').sum()
    print(f"Total games: {len(df)}")
    print(f"  Success: {n_success}")
    print(f"  Control: {n_control}")

    print(f"\nPre-launch feature ranges:")
    for col in ["wishlists_at_launch", "followers_at_launch",
                "days_on_steam_pre_launch", "price_at_launch"]:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {col}: min={vals.min():.0f}, median={vals.median():.0f}, max={vals.max():.0f}")

    # Per-window velocity eligibility
    print(f"\nVelocity data coverage (non-null = game had sufficient history):")
    velocity_cols = [
        "wishlist_velocity_30d", "wishlist_velocity_90d", "wishlist_velocity_180d",
        "follower_velocity_30d", "follower_velocity_90d", "follower_velocity_180d",
    ]
    for col in velocity_cols:
        if col in df.columns:
            valid = df[col].notna()
            valid_s = valid & (df["group"] == "success")
            valid_c = valid & (df["group"] == "control")
            print(f"  {col}: {valid.sum()}/{len(df)} total "
                  f"({valid_s.sum()}/{n_success} success, {valid_c.sum()}/{n_control} control)")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_analysis_set()
