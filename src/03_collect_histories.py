"""Phase 3: Collect full history for success games + control group."""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from api_client import GamalyticClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORIES_DIR = DATA_DIR / "histories"
CONFIG_DIR = PROJECT_ROOT / "config"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
CONTROL_FILE = CONFIG_DIR / "control_group.json"
TIMESERIES_FILE = DATA_DIR / "timeseries.parquet"

DETAIL_FIELDS = (
    "history,tags,genres,features,price,developers,publishers,"
    "copiesSold,revenue,totalRevenue,wishlists,followers,reviews,"
    "reviewScore,firstReleaseDate,earlyAccessExitDate,releaseDate,"
    "EAReleaseDate,earlyAccess"
)


def get_all_game_ids() -> list[dict]:
    """Collect all game IDs we need histories for (success + control)."""
    games = []

    success_games = json.loads(SUCCESS_FILE.read_text())
    for sg in success_games:
        games.append({"steamId": str(sg["steamId"]), "name": sg["name"], "group": "success"})

    if CONTROL_FILE.exists():
        control_group = json.loads(CONTROL_FILE.read_text())
        control_ids = set()
        for matches in control_group.values():
            for cid in matches:
                control_ids.add(str(cid))

        for cid in control_ids:
            games.append({"steamId": cid, "name": None, "group": "control"})

    return games


def collect_histories():
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    client = GamalyticClient()

    games = get_all_game_ids()
    already_done = {p.stem for p in HISTORIES_DIR.glob("*.json")}

    remaining = [g for g in games if g["steamId"] not in already_done]
    total = len(games)
    done = total - len(remaining)

    logger.info(f"Total games to collect: {total}")
    logger.info(f"Already collected: {done}")
    logger.info(f"Remaining: {len(remaining)}")

    start_time = time.time()

    for i, game in enumerate(remaining):
        sid = game["steamId"]
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 and i > 0 else 0
        eta = (len(remaining) - i) / rate if rate > 0 else 0

        label = game["name"] or sid
        logger.info(
            f"[{done + i + 1}/{total}] Fetching {label} ({game['group']}) "
            f"[ETA: {eta/60:.1f}min] [{client.stats}]"
        )

        data = client.get_game_details(
            sid,
            fields=DETAIL_FIELDS,
            include_pre_release_history=True,
        )

        if data is None:
            logger.warning(f"Failed to fetch {sid}, skipping.")
            continue

        # Save raw JSON
        out_path = HISTORIES_DIR / f"{sid}.json"
        out_path.write_text(json.dumps(data, indent=2))

        # Fill in name for control games
        if game["name"] is None and "name" in data:
            game["name"] = data["name"]

    # Build combined timeseries parquet
    logger.info("Building combined timeseries parquet...")
    build_timeseries()
    logger.info(f"Done. {client.stats}")


def build_timeseries():
    """Combine all history JSON files into a single timeseries parquet."""
    rows = []

    for json_file in HISTORIES_DIR.glob("*.json"):
        steam_id = json_file.stem
        data = json.loads(json_file.read_text())
        history = data.get("history", [])

        if not history:
            continue

        for entry in history:
            rows.append({
                "steamId": steam_id,
                "timestamp": entry.get("timeStamp"),
                "reviews": entry.get("reviews"),
                "price": entry.get("price"),
                "score": entry.get("score"),
                "players": entry.get("players"),
                "avgPlaytime": entry.get("avgPlaytime"),
                "sales": entry.get("sales"),
                "revenue": entry.get("revenue"),
                "followers": entry.get("followers"),
                "wishlists": entry.get("wishlists"),
            })

    if not rows:
        logger.warning("No history data found!")
        return

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["steamId"] = df["steamId"].astype(str)
    df = df.sort_values(["steamId", "timestamp"])
    df.to_parquet(TIMESERIES_FILE, index=False)
    logger.info(f"Saved {len(df):,} time series rows for {df['steamId'].nunique()} games to {TIMESERIES_FILE}")


if __name__ == "__main__":
    collect_histories()
