"""Collect history for a stratified random sample of backdrop games.

Extends the histories/ directory with a representative sample from the
full backdrop population (61K+ games), so strategy evaluations can be
computed against a realistic screening pool rather than just 300 controls.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from api_client import GamalyticClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORIES_DIR = DATA_DIR / "histories"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"

DETAIL_FIELDS = (
    "history,tags,genres,features,price,developers,publishers,"
    "copiesSold,revenue,totalRevenue,wishlists,followers,reviews,"
    "reviewScore,firstReleaseDate,earlyAccessExitDate,releaseDate,"
    "EAReleaseDate,earlyAccess"
)

# Stratified sampling config: (follower_min, follower_max, sample_size_or_all)
# "all" means take every game in that stratum
STRATA = [
    (0, 100, 200),
    (100, 500, 200),
    (500, 1000, 200),
    (1000, 3000, 300),
    (3000, 5000, "all"),
    (5000, 10000, "all"),
    (10000, 30000, "all"),
    (30000, 100000, "all"),
    (100000, float("inf"), "all"),
]


def build_sample(bd: pd.DataFrame, existing_ids: set) -> pd.DataFrame:
    """Build stratified sample from backdrop, excluding already-collected games."""
    bd = bd[~bd["steamId"].isin(existing_ids)].copy()

    samples = []
    for fol_min, fol_max, n in STRATA:
        stratum = bd[(bd["followers"] >= fol_min) & (bd["followers"] < fol_max)]
        if n == "all":
            chosen = stratum
        else:
            chosen = stratum.sample(n=min(n, len(stratum)), random_state=42)
        samples.append(chosen)
        logger.info(f"  Stratum [{fol_min:>6,}-{fol_max:>10}): {len(stratum):>6,} games, sampled {len(chosen):>5,}")

    result = pd.concat(samples, ignore_index=True)
    logger.info(f"  Total sample: {len(result):,} games")
    return result


def collect_histories(sample_df: pd.DataFrame):
    """Collect game detail histories for sampled games."""
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    client = GamalyticClient()

    already_done = {p.stem for p in HISTORIES_DIR.glob("*.json")}
    to_fetch = sample_df[~sample_df["steamId"].isin(already_done)]

    logger.info(f"Need to fetch: {len(to_fetch)} (skipping {len(sample_df) - len(to_fetch)} already done)")

    success, fail = 0, 0
    total = len(to_fetch)

    for i, (_, row) in enumerate(to_fetch.iterrows()):
        sid = row["steamId"]
        try:
            data = client.get_game_details(sid, fields=DETAIL_FIELDS)
            if data:
                out_path = HISTORIES_DIR / f"{sid}.json"
                out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                success += 1
            else:
                fail += 1
        except Exception as e:
            logger.warning(f"  Failed {sid}: {e}")
            fail += 1

        if (i + 1) % 100 == 0 or i + 1 == total:
            logger.info(f"  Progress: {i+1}/{total} (success={success}, fail={fail})")

    logger.info(f"Done: {success} collected, {fail} failed")
    return success, fail


def main():
    logger.info("Loading backdrop data...")
    bd = pd.read_parquet(BACKDROP_FILE)
    bd["steamId"] = bd["steamId"].astype(str)

    existing_ids = {p.stem for p in HISTORIES_DIR.glob("*.json")}
    logger.info(f"Existing history files: {len(existing_ids)}")

    logger.info("Building stratified sample...")
    sample = build_sample(bd, existing_ids)

    logger.info("Collecting histories from API...")
    collect_histories(sample)

    final_count = len(list(HISTORIES_DIR.glob("*.json")))
    logger.info(f"Total history files now: {final_count}")


if __name__ == "__main__":
    main()
