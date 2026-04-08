"""Phase 2: Match control group games to success games using tag similarity."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
CONTROL_FILE = CONFIG_DIR / "control_group.json"


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def date_proximity_score(date_a_ms: float, date_b_ms: float, max_days: int = 180) -> float:
    """Score from 0-1 based on how close two dates are (within max_days)."""
    if pd.isna(date_a_ms) or pd.isna(date_b_ms):
        return 0.0
    diff_days = abs(date_a_ms - date_b_ms) / (1000 * 60 * 60 * 24)
    if diff_days > max_days:
        return 0.0
    return 1.0 - (diff_days / max_days)


def price_proximity_score(price_a: float, price_b: float, max_diff: float = 10.0) -> float:
    """Score from 0-1 based on how close two prices are (within max_diff)."""
    if pd.isna(price_a) or pd.isna(price_b):
        return 0.5  # neutral if unknown
    diff = abs(price_a - price_b)
    if diff > max_diff:
        return 0.0
    return 1.0 - (diff / max_diff)


def _to_tag_set(tags) -> set:
    if tags is None:
        return set()
    if isinstance(tags, np.ndarray):
        return set(tags.tolist())
    return set(tags)


def compute_composite_score(success_game: dict, candidate: dict) -> float:
    """Compute weighted composite similarity score."""
    success_tags = _to_tag_set(success_game.get("tags"))
    candidate_tags = _to_tag_set(candidate.get("tags"))
    tag_score = jaccard_similarity(success_tags, candidate_tags)

    date_score = date_proximity_score(
        success_game.get("firstReleaseDate_ms", 0),
        candidate.get("firstReleaseDate_ms", 0),
    )

    price_score = price_proximity_score(
        success_game.get("price", 0),
        candidate.get("price", 0),
    )

    return 0.6 * tag_score + 0.2 * date_score + 0.2 * price_score


def format_tags(tags, max_shown: int = 5) -> str:
    if tags is None or (hasattr(tags, '__len__') and len(tags) == 0):
        return "(none)"
    tag_list = list(tags) if isinstance(tags, np.ndarray) else list(tags)
    shown = tag_list[:max_shown]
    extra = len(tag_list) - max_shown
    result = ", ".join(str(t) for t in shown)
    if extra > 0:
        result += f" +{extra} more"
    return result


def format_revenue(rev) -> str:
    if pd.isna(rev) or rev is None:
        return "N/A"
    if rev >= 1_000_000:
        return f"${rev/1_000_000:.1f}M"
    if rev >= 1_000:
        return f"${rev/1_000:.0f}K"
    return f"${rev:.0f}"


def format_date(dt) -> str:
    if pd.isna(dt):
        return "N/A"
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt / 1000)
    return dt.strftime("%Y-%m-%d")


def interactive_confirm(success_game: dict, candidates: list[dict]) -> list[str]:
    """Present candidates and let user confirm/edit the selection."""
    print(f"\n{'=' * 70}")
    print(f"SUCCESS GAME: {success_game['name']} (ID: {success_game['steamId']})")
    print(f"Tags: {format_tags(success_game.get('tags', []), 10)}")
    print(f"Price: ${success_game.get('price', 'N/A')} | "
          f"Release: {format_date(success_game.get('firstReleaseDate_ms', None))}")
    print(f"{'=' * 70}")
    print(f"\nTop {len(candidates)} matched candidates:\n")
    print(f"{'#':>3} {'Name':<35} {'Score':>5} {'Revenue':>10} {'Release':>12} Tags")
    print("-" * 100)

    for i, c in enumerate(candidates, 1):
        print(f"{i:>3} {c['name'][:34]:<35} {c['score']:.3f} "
              f"{format_revenue(c.get('revenue')):>10} "
              f"{format_date(c.get('firstReleaseDate_ms')):>12} "
              f"{format_tags(c.get('tags', []), 3)}")

    print(f"\nOptions:")
    print(f"  y        - accept all {len(candidates)}")
    print(f"  n 3,7    - remove candidates #3 and #7")
    print(f"  a <id>   - manually add a game by Steam ID")
    print(f"  s        - skip this success game")

    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == "y":
            return [c["steamId"] for c in candidates]

        if choice == "s":
            return []

        if choice.startswith("n "):
            try:
                remove_indices = {int(x.strip()) for x in choice[2:].split(",")}
                kept = [c for i, c in enumerate(candidates, 1) if i not in remove_indices]
                print(f"Keeping {len(kept)} candidates: {', '.join(c['name'] for c in kept)}")
                confirm = input("Confirm? (y/n): ").strip().lower()
                if confirm == "y":
                    return [c["steamId"] for c in kept]
            except ValueError:
                print("Invalid format. Use: n 3,7")

        if choice.startswith("a "):
            steam_id = choice[2:].strip()
            existing = [c["steamId"] for c in candidates]
            existing.append(steam_id)
            print(f"Added Steam ID {steam_id}. Total: {len(existing)} candidates.")
            return existing

        print("Invalid choice. Try again.")


def match_control_group():
    # Load data
    if not BACKDROP_FILE.exists():
        print("Error: Run 01_collect_backdrop.py first to create backdrop_games.parquet")
        sys.exit(1)

    df = pd.read_parquet(BACKDROP_FILE)
    success_games = json.loads(SUCCESS_FILE.read_text())
    success_ids = {str(g["steamId"]) for g in success_games}

    # Load existing control group if present
    existing_control = {}
    if CONTROL_FILE.exists():
        existing_control = json.loads(CONTROL_FILE.read_text())

    # Ensure steamId is string
    df["steamId"] = df["steamId"].astype(str)

    # Store original ms timestamps for scoring
    if "firstReleaseDate" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["firstReleaseDate"]):
            # Force to ns resolution first, then convert to ms — safe regardless of source dtype
            df["firstReleaseDate_ms"] = df["firstReleaseDate"].astype("datetime64[ns]").astype("int64") // 10**6
        else:
            df["firstReleaseDate_ms"] = df["firstReleaseDate"]

    # Exclude success games and games with no tags from candidate pool
    def has_tags(x):
        return isinstance(x, (list, np.ndarray)) and len(x) > 0

    candidates_df = df[
        (~df["steamId"].isin(success_ids)) &
        (df["tags"].apply(has_tags))
    ].copy()

    print(f"Loaded {len(df):,} backdrop games, {len(candidates_df):,} eligible candidates")
    print(f"Success games: {len(success_games)}")

    control_group = {}

    for sg in success_games:
        sg_id = str(sg["steamId"])

        # Check if already matched
        if sg_id in existing_control:
            reuse = input(f"\n{sg['name']} already has {len(existing_control[sg_id])} matches. "
                          f"Keep existing? (y/n): ").strip().lower()
            if reuse == "y":
                control_group[sg_id] = existing_control[sg_id]
                continue

        # Get success game details from backdrop
        sg_row = df[df["steamId"] == sg_id]
        if sg_row.empty:
            print(f"\nWarning: {sg['name']} (ID: {sg_id}) not found in backdrop data.")
            print("It may not be released yet. Skipping matching (you can add controls manually).")
            continue

        sg_data = sg_row.iloc[0].to_dict()

        # Fully vectorized scoring using explode + groupby
        sg_tags = _to_tag_set(sg_data.get("tags"))
        sg_date_ms = sg_data.get("firstReleaseDate_ms", 0)
        sg_price = sg_data.get("price", 0)
        sg_tag_count = len(sg_tags)

        print(f"  Scoring {len(candidates_df):,} candidates...")

        # Tag count per candidate
        c_tag_counts = candidates_df["tags"].apply(len)

        # Intersection size via explode: only count tags that are in sg_tags
        exploded = candidates_df[["steamId", "tags"]].explode("tags")
        shared = exploded[exploded["tags"].isin(sg_tags)]
        intersection_counts = shared.groupby("steamId").size()

        # Jaccard = |A∩B| / |A∪B| = |A∩B| / (|A| + |B| - |A∩B|)
        isect = candidates_df["steamId"].map(intersection_counts).fillna(0)
        union = sg_tag_count + c_tag_counts - isect
        tag_score = (isect / union).fillna(0)

        # Date proximity (vectorized)
        if pd.notna(sg_date_ms) and sg_date_ms != 0:
            diff_days = (candidates_df["firstReleaseDate_ms"] - sg_date_ms).abs() / 86_400_000
            date_score = (1.0 - diff_days / 180).clip(lower=0)
        else:
            date_score = 0.0

        # Price proximity (vectorized)
        if pd.notna(sg_price):
            price_diff = (candidates_df["price"] - sg_price).abs()
            price_score = (1.0 - price_diff / 10).clip(lower=0).fillna(0.5)
        else:
            price_score = 0.5

        composite = 0.6 * tag_score + 0.2 * date_score + 0.2 * price_score

        # Top 10 by composite score
        top_idx = composite.nlargest(10).index
        top_df = candidates_df.loc[top_idx].copy()
        top_df["_score"] = composite.loc[top_idx]

        top_candidates = []
        for _, row in top_df.iterrows():
            if row["_score"] > 0.05:
                tags = row.get("tags")
                top_candidates.append({
                    "steamId": row["steamId"],
                    "name": row["name"],
                    "score": row["_score"],
                    "tags": tags.tolist() if isinstance(tags, np.ndarray) else tags,
                    "revenue": row.get("revenue"),
                    "price": row.get("price"),
                    "firstReleaseDate_ms": row.get("firstReleaseDate_ms"),
                })

        if not top_candidates:
            print(f"\nNo suitable matches found for {sg['name']}.")
            continue

        # Interactive confirmation
        selected_ids = interactive_confirm(sg_data, top_candidates)
        if selected_ids:
            control_group[sg_id] = selected_ids
            print(f"Saved {len(selected_ids)} controls for {sg['name']}")

    # Save control group
    CONTROL_FILE.write_text(json.dumps(control_group, indent=2))
    print(f"\nControl group saved to {CONTROL_FILE}")
    total_controls = sum(len(v) for v in control_group.values())
    print(f"Total: {len(control_group)} success games with {total_controls} control matches")


if __name__ == "__main__":
    match_control_group()
