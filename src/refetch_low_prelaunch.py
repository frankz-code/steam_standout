"""Re-fetch history for games with <30 days pre-launch data, using include_pre_release_history=true.

Compares old vs new history coverage and saves updated files when more data is available.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from api_client import GamalyticClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data"
HISTORIES_DIR = DATA_DIR / "histories"
CONFIG_DIR = PROJECT / "config"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
CONTROL_FILE = CONFIG_DIR / "control_group.json"

DETAIL_FIELDS = (
    "history,tags,genres,features,price,developers,publishers,"
    "copiesSold,revenue,totalRevenue,wishlists,followers,reviews,"
    "reviewScore,firstReleaseDate,earlyAccessExitDate,releaseDate,"
    "EAReleaseDate,earlyAccess"
)


def ts_to_date(ts_ms):
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, ValueError, TypeError):
        return None


def get_history_stats(data):
    """Extract history coverage stats from a game data dict."""
    history = data.get("history", [])
    if not history:
        return {"entries": 0, "earliest": None, "pre_launch_days": None}

    timestamps = [h.get("timeStamp") for h in history if h.get("timeStamp")]
    if not timestamps:
        return {"entries": len(history), "earliest": None, "pre_launch_days": None}

    earliest = min(timestamps)
    release_ts = data.get("firstReleaseDate") or data.get("releaseDate")

    pre_launch_days = None
    if release_ts:
        release_dt = datetime.fromtimestamp(release_ts / 1000, tz=timezone.utc)
        earliest_dt = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc)
        pre_launch_days = (release_dt - earliest_dt).days

    return {
        "entries": len(history),
        "earliest": earliest,
        "earliest_date": ts_to_date(earliest),
        "pre_launch_days": pre_launch_days,
    }


def find_low_prelaunch_games():
    """Identify all games with <30 days pre-launch coverage."""
    success_games = json.loads(SUCCESS_FILE.read_text())
    success_ids = {str(g["steamId"]): g.get("name", "") for g in success_games}

    control_map = json.loads(CONTROL_FILE.read_text())
    control_ids = set()
    for matches in control_map.values():
        for cid in matches:
            control_ids.add(str(cid))

    all_ids = {}
    for sid, name in success_ids.items():
        all_ids[sid] = {"name": name, "group": "success"}
    for cid in control_ids:
        all_ids[cid] = {"name": None, "group": "control"}

    low_prelaunch = []

    for steam_id, info in all_ids.items():
        path = HISTORIES_DIR / f"{steam_id}.json"
        if not path.exists():
            continue

        data = json.loads(path.read_text())
        if info["name"] is None:
            info["name"] = data.get("name", steam_id)

        stats = get_history_stats(data)

        if stats["pre_launch_days"] is not None and stats["pre_launch_days"] < 30:
            low_prelaunch.append({
                "steamId": steam_id,
                "name": info["name"],
                "group": info["group"],
                "old_entries": stats["entries"],
                "old_earliest": stats["earliest_date"],
                "old_pre_launch_days": stats["pre_launch_days"],
            })

    return low_prelaunch


def refetch_games():
    games = find_low_prelaunch_games()

    print(f"\n{'='*80}")
    print(f"RE-FETCHING {len(games)} GAMES WITH LOW PRE-LAUNCH COVERAGE")
    print(f"{'='*80}")
    print(f"{'ID':>10}  {'Name':<35}  {'Group':<8}  {'Old Days':>8}  {'Old Entries':>11}")
    print("-" * 80)
    for g in sorted(games, key=lambda x: x["old_pre_launch_days"]):
        print(f"{g['steamId']:>10}  {(g['name'] or '?')[:34]:<35}  {g['group']:<8}  "
              f"{g['old_pre_launch_days']:>8}  {g['old_entries']:>11}")

    client = GamalyticClient()
    results = []
    updated = 0
    unchanged = 0

    for i, game in enumerate(games):
        sid = game["steamId"]
        logger.info(f"[{i+1}/{len(games)}] Fetching {game['name'] or sid} ({game['group']})...")

        new_data = client.get_game_details(
            sid,
            fields=DETAIL_FIELDS,
            include_pre_release_history=True,
        )

        if new_data is None:
            logger.warning(f"  Failed to fetch {sid}, skipping")
            results.append({**game, "status": "FETCH_FAILED"})
            continue

        new_stats = get_history_stats(new_data)

        result = {
            **game,
            "new_entries": new_stats["entries"],
            "new_earliest": new_stats.get("earliest_date"),
            "new_pre_launch_days": new_stats["pre_launch_days"],
        }

        # Did we get more data?
        gained_entries = new_stats["entries"] - game["old_entries"]
        gained_days = (new_stats["pre_launch_days"] or 0) - (game["old_pre_launch_days"] or 0)

        if gained_entries > 0 or gained_days > 0:
            result["status"] = f"IMPROVED (+{gained_entries} entries, +{gained_days} days)"
            # Save updated file
            out_path = HISTORIES_DIR / f"{sid}.json"
            out_path.write_text(json.dumps(new_data, indent=2))
            updated += 1
            logger.info(f"  IMPROVED: {game['old_entries']}→{new_stats['entries']} entries, "
                        f"{game['old_pre_launch_days']}→{new_stats['pre_launch_days']} pre-launch days. Saved.")
        else:
            result["status"] = "NO_CHANGE"
            unchanged += 1
            logger.info(f"  No change: {new_stats['entries']} entries, "
                        f"{new_stats['pre_launch_days']} pre-launch days")

        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("REFETCH RESULTS")
    print(f"{'='*80}")
    print(f"\n{'ID':>10}  {'Name':<30}  {'Grp':<5}  {'OldD':>5}  {'NewD':>5}  {'OldE':>5}  {'NewE':>5}  Status")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x.get("new_pre_launch_days") or -999):
        print(f"{r['steamId']:>10}  {(r['name'] or '?')[:29]:<30}  {r['group'][:5]:<5}  "
              f"{r['old_pre_launch_days']:>5}  "
              f"{r.get('new_pre_launch_days', '?'):>5}  "
              f"{r['old_entries']:>5}  "
              f"{r.get('new_entries', '?'):>5}  "
              f"{r['status']}")

    print(f"\nTotal: {len(games)} games")
    print(f"  Updated (more data): {updated}")
    print(f"  Unchanged: {unchanged}")
    print(f"  Failed: {len(games) - updated - unchanged}")
    print(f"\nAPI stats: {client.stats}")


if __name__ == "__main__":
    refetch_games()
