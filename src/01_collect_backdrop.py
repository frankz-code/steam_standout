"""Phase 1: Collect all Steam games released 2023-01-01 to 2026-03-18 as backdrop dataset."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from api_client import GamalyticClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROGRESS_FILE = DATA_DIR / "backdrop_progress.json"
OUTPUT_FILE = DATA_DIR / "backdrop_games.parquet"

FIELDS = (
    "steamId,name,price,reviews,reviewScore,followers,wishlists,"
    "copiesSold,revenue,totalRevenue,tags,genres,features,"
    "developers,publishers,firstReleaseDate,earlyAccess,earlyAccessExitDate"
)

# Date range: 2023-01-01 to 2026-03-18 (Unix epoch milliseconds)
DATE_MIN = int(datetime(2023, 1, 1).timestamp() * 1000)
DATE_MAX = int(datetime(2026, 3, 18).timestamp() * 1000)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"pages_collected": [], "all_games": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress))


def collect_backdrop():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = GamalyticClient()
    progress = load_progress()

    pages_done = set(progress["pages_collected"])
    all_games = progress["all_games"]
    seen_ids = {g["steamId"] for g in all_games}

    page = 0
    consecutive_empty = 0

    logger.info(f"Starting backdrop collection. {len(all_games)} games already collected.")
    logger.info(f"Date range: {datetime.fromtimestamp(DATE_MIN/1000)} to {datetime.fromtimestamp(DATE_MAX/1000)}")

    while consecutive_empty < 3:
        if page in pages_done:
            page += 1
            consecutive_empty = 0
            continue

        logger.info(f"Fetching page {page}... ({len(all_games)} games so far) [{client.stats}]")

        result = client.get_game_list(
            page=page,
            limit=1000,
            sort="id",
            sort_mode="asc",
            fields=FIELDS,
            first_release_date_min=DATE_MIN,
            first_release_date_max=DATE_MAX,
            release_status="all",
        )

        if result is None:
            logger.error(f"Failed to fetch page {page}, stopping.")
            break

        games = result if isinstance(result, list) else result.get("result", result.get("data", []))

        if not games:
            consecutive_empty += 1
            logger.info(f"Empty page {page} (consecutive empty: {consecutive_empty})")
            page += 1
            continue

        consecutive_empty = 0
        new_count = 0
        for game in games:
            sid = str(game.get("steamId", ""))
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                all_games.append(game)
                new_count += 1

        logger.info(f"Page {page}: {len(games)} returned, {new_count} new. Total: {len(all_games)}")

        pages_done.add(page)
        progress["pages_collected"] = sorted(pages_done)
        progress["all_games"] = all_games
        save_progress(progress)

        page += 1

    # Build and save parquet
    if not all_games:
        logger.warning("No games collected!")
        return

    df = pd.DataFrame(all_games)

    # Convert epoch ms to datetime
    for col in ["firstReleaseDate", "earlyAccessExitDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")

    # Ensure steamId is string
    df["steamId"] = df["steamId"].astype(str)

    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"\nSaved {len(df)} games to {OUTPUT_FILE}")

    # Summary stats
    print("\n" + "=" * 60)
    print("BACKDROP COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total games: {len(df):,}")
    if "firstReleaseDate" in df.columns:
        print(f"Date range: {df['firstReleaseDate'].min()} to {df['firstReleaseDate'].max()}")
    if "revenue" in df.columns:
        rev = df["revenue"].dropna()
        print(f"Revenue: median=${rev.median():,.0f}, mean=${rev.mean():,.0f}, max=${rev.max():,.0f}")
    if "reviews" in df.columns:
        print(f"Reviews: median={df['reviews'].dropna().median():,.0f}")
    if "tags" in df.columns:
        print(f"Games with tags: {df['tags'].apply(lambda x: hasattr(x, '__len__') and len(x) > 0).sum():,}")
    print(f"\nAPI stats: {client.stats}")


if __name__ == "__main__":
    collect_backdrop()
