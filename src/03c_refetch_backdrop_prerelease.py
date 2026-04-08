"""Re-fetch backdrop histories with include_pre_release_history=True.

The original 03b_collect_backdrop_sample.py omitted this flag, causing
most backdrop games to lack pre-launch history data (only 748/4872 had it).
This script re-fetches games that currently lack pre-launch data, saving
updated JSON files when more history is obtained.

Follows the pattern of refetch_low_prelaunch.py.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from api_client import GamalyticClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
HISTORIES_DIR = DATA_DIR / "histories"
CONFIG_DIR = PROJECT / "config"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
CONTROL_FILE = CONFIG_DIR / "control_group.json"

DETAIL_FIELDS = (
    "history,tags,genres,features,price,developers,publishers,"
    "copiesSold,revenue,totalRevenue,wishlists,followers,reviews,"
    "reviewScore,firstReleaseDate,earlyAccessExitDate,releaseDate,"
    "EAReleaseDate,earlyAccess"
)

# Only re-fetch games with >= this many followers (smaller games unlikely
# to have been tracked pre-launch by Gamalytic)
MIN_FOLLOWERS_FOR_REFETCH = 500


def get_prelaunch_days(data):
    """Return number of pre-launch days in history, or None."""
    history = data.get("history", [])
    if not history:
        return None

    release_ms = data.get("releaseDate") or data.get("firstReleaseDate")
    if not release_ms:
        return None

    timestamps = [h.get("timeStamp") or h.get("date") for h in history]
    timestamps = [t for t in timestamps if t is not None]
    if not timestamps:
        return None

    earliest = min(timestamps)
    release_dt = datetime.fromtimestamp(release_ms / 1000, tz=timezone.utc)
    earliest_dt = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc)
    return (release_dt - earliest_dt).days


def find_candidates():
    """Find backdrop games that need re-fetching."""
    # Load success + control IDs to exclude them
    with open(SUCCESS_FILE) as f:
        success_ids = {str(g["steamId"]) for g in json.load(f)}
    with open(CONTROL_FILE) as f:
        control_map = json.load(f)
    control_ids = set()
    for matches in control_map.values():
        for cid in matches:
            control_ids.add(str(cid))
    exclude_ids = success_ids | control_ids

    # Load backdrop for follower filter
    bd = pd.read_parquet(BACKDROP_FILE)
    bd["steamId"] = bd["steamId"].astype(str)
    high_followers = set(bd[bd["followers"] >= MIN_FOLLOWERS_FOR_REFETCH]["steamId"])

    candidates = []
    already_has_prelaunch = 0
    too_few_followers = 0
    is_success_control = 0

    for hf in sorted(HISTORIES_DIR.glob("*.json")):
        sid = hf.stem

        # Skip success/control games (already fetched with the flag)
        if sid in exclude_ids:
            is_success_control += 1
            continue

        # Skip low-follower games
        if sid not in high_followers:
            too_few_followers += 1
            continue

        try:
            data = json.loads(hf.read_text(encoding="utf-8"))
        except Exception:
            continue

        pre_days = get_prelaunch_days(data)
        if pre_days is not None and pre_days > 7:
            already_has_prelaunch += 1
            continue

        old_entries = len(data.get("history", []))
        name = data.get("name", sid)
        candidates.append({
            "steamId": sid, "name": name,
            "old_entries": old_entries,
            "old_pre_days": pre_days,
        })

    logger.info(f"Scan complete:")
    logger.info(f"  Success/control (skipped): {is_success_control}")
    logger.info(f"  Low followers < {MIN_FOLLOWERS_FOR_REFETCH} (skipped): {too_few_followers}")
    logger.info(f"  Already has pre-launch > 7d: {already_has_prelaunch}")
    logger.info(f"  Candidates for re-fetch: {len(candidates)}")

    return candidates


def refetch(candidates):
    """Re-fetch candidates with include_pre_release_history=True."""
    client = GamalyticClient()
    updated = 0
    unchanged = 0
    failed = 0

    total = len(candidates)
    for i, game in enumerate(candidates):
        sid = game["steamId"]

        try:
            new_data = client.get_game_details(
                sid,
                fields=DETAIL_FIELDS,
                include_pre_release_history=True,
            )
        except Exception as e:
            logger.warning(f"  [{i+1}/{total}] Failed {sid}: {e}")
            failed += 1
            continue

        if new_data is None:
            failed += 1
            continue

        new_entries = len(new_data.get("history", []))
        new_pre_days = get_prelaunch_days(new_data)
        gained = new_entries - game["old_entries"]

        if gained > 0 or (new_pre_days or 0) > (game["old_pre_days"] or 0):
            out_path = HISTORIES_DIR / f"{sid}.json"
            out_path.write_text(json.dumps(new_data, ensure_ascii=False), encoding="utf-8")
            updated += 1
            if (i + 1) % 50 == 0 or gained > 10:
                logger.info(f"  [{i+1}/{total}] {game['name'][:30]}: "
                            f"+{gained} entries, pre-launch {game['old_pre_days']}→{new_pre_days} days")
        else:
            unchanged += 1

        if (i + 1) % 200 == 0:
            logger.info(f"  Progress: {i+1}/{total} (updated={updated}, unchanged={unchanged}, failed={failed})")

    logger.info(f"\nDone: {updated} updated, {unchanged} unchanged, {failed} failed")
    logger.info(f"API stats: {client.stats}")
    return updated, unchanged, failed


def main():
    logger.info("Finding candidates for re-fetch...")
    candidates = find_candidates()

    if not candidates:
        logger.info("No candidates found. All backdrop games already have pre-launch data or are below follower threshold.")
        return

    logger.info(f"\nRe-fetching {len(candidates)} games with include_pre_release_history=True...")
    updated, unchanged, failed = refetch(candidates)

    # Count final pre-launch coverage
    has_prelaunch = 0
    total_files = 0
    for hf in HISTORIES_DIR.glob("*.json"):
        total_files += 1
        try:
            data = json.loads(hf.read_text(encoding="utf-8"))
            pre_days = get_prelaunch_days(data)
            if pre_days is not None and pre_days > 0:
                has_prelaunch += 1
        except Exception:
            pass

    logger.info(f"\nFinal coverage: {has_prelaunch}/{total_files} files have pre-launch data "
                f"({has_prelaunch/total_files:.1%})")


if __name__ == "__main__":
    main()
