#!/usr/bin/env python3
"""
Comprehensive data completeness audit for the Steam Standout project.

Checks:
1. Missing history files for success and control games
2. History coverage quality (pre-launch days, followers/wishlists completeness)
3. Flags problem games with specific issues
4. Summarizes timeseries and backdrop coverage
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT = Path("/Users/admin/ClaudeProjects/steam_standout")
SUCCESS_PATH = PROJECT / "config" / "success_games.json"
CONTROL_PATH = PROJECT / "config" / "control_group.json"
HISTORIES_DIR = PROJECT / "data" / "histories"
TIMESERIES_PATH = PROJECT / "data" / "timeseries.parquet"
BACKDROP_PATH = PROJECT / "data" / "backdrop_games.parquet"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def ts_to_date(ts_ms):
    """Convert a millisecond timestamp to a date string, or None."""
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, ValueError, TypeError):
        return None


def ts_to_dt(ts_ms):
    """Convert a millisecond timestamp to a datetime, or None."""
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    except (OSError, ValueError, TypeError):
        return None


def audit_history_file(steam_id, game_name=None):
    """Audit a single history JSON file. Returns a dict of findings."""
    path = HISTORIES_DIR / f"{steam_id}.json"
    result = {
        "steamId": steam_id,
        "name": game_name or "(unknown)",
        "file_exists": path.exists(),
        "history_exists": False,
        "history_length": 0,
        "earliest_timestamp": None,
        "earliest_date": None,
        "first_release_date": None,
        "release_date_str": None,
        "pre_launch_days": None,
        "followers_non_null_pct": 0.0,
        "wishlists_non_null_pct": 0.0,
        "issues": [],
    }

    if not path.exists():
        result["issues"].append("MISSING_FILE")
        return result

    with open(path) as f:
        data = json.load(f)

    # --- Release date ---
    release_ts = data.get("firstReleaseDate") or data.get("releaseDate")
    if release_ts:
        result["first_release_date"] = release_ts
        result["release_date_str"] = ts_to_date(release_ts)
    else:
        result["issues"].append("NO_RELEASE_DATE")

    # --- History array ---
    history = data.get("history")
    if not history or not isinstance(history, list) or len(history) == 0:
        result["issues"].append("EMPTY_HISTORY")
        return result

    result["history_exists"] = True
    result["history_length"] = len(history)

    # Earliest / latest timestamp
    timestamps = [h.get("timeStamp") for h in history if h.get("timeStamp")]
    if timestamps:
        earliest = min(timestamps)
        result["earliest_timestamp"] = earliest
        result["earliest_date"] = ts_to_date(earliest)

        # Pre-launch days
        if release_ts:
            release_dt = ts_to_dt(release_ts)
            earliest_dt = ts_to_dt(earliest)
            if release_dt and earliest_dt:
                delta = (release_dt - earliest_dt).days
                result["pre_launch_days"] = delta
                if delta < 30:
                    result["issues"].append(f"LOW_PRE_LAUNCH ({delta}d)")

    # Followers coverage
    total = len(history)
    followers_non_null = sum(
        1 for h in history
        if h.get("followers") is not None
    )
    wishlists_non_null = sum(
        1 for h in history
        if h.get("wishlists") is not None
    )
    result["followers_non_null_pct"] = round(100 * followers_non_null / total, 1)
    result["wishlists_non_null_pct"] = round(100 * wishlists_non_null / total, 1)

    if result["followers_non_null_pct"] < 50:
        result["issues"].append(f"LOW_FOLLOWERS ({result['followers_non_null_pct']}%)")
    if result["wishlists_non_null_pct"] < 50:
        result["issues"].append(f"LOW_WISHLISTS ({result['wishlists_non_null_pct']}%)")

    return result


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #

def main():
    # ---- Load configs ----
    with open(SUCCESS_PATH) as f:
        success_games = json.load(f)
    success_ids = {g["steamId"]: g.get("name", "") for g in success_games}

    with open(CONTROL_PATH) as f:
        control_map = json.load(f)

    # Flatten all control IDs (skip special keys like _manual, _fill)
    control_ids = set()
    control_to_success = {}  # map control_id -> success_id for context
    for key, ids in control_map.items():
        for cid in ids:
            control_ids.add(cid)
            if key not in ("_manual", "_fill"):
                control_to_success[cid] = key

    # Also include _manual and _fill IDs
    for key in ("_manual", "_fill"):
        if key in control_map:
            for cid in control_map[key]:
                control_ids.add(cid)

    # ---- Existing history files ----
    existing_files = {
        p.stem for p in HISTORIES_DIR.glob("*.json")
    }

    print("=" * 80)
    print("STEAM STANDOUT DATA COMPLETENESS AUDIT")
    print("=" * 80)
    print()

    # ---- 1. Missing files ----
    missing_success = [sid for sid in success_ids if sid not in existing_files]
    missing_control = [cid for cid in control_ids if cid not in existing_files]

    print("-" * 80)
    print("1. MISSING HISTORY FILES")
    print("-" * 80)
    print(f"   Success games total: {len(success_ids)}")
    print(f"   Success games missing files: {len(missing_success)}")
    if missing_success:
        for sid in missing_success:
            print(f"      - {sid} ({success_ids.get(sid, '?')})")

    print(f"   Control games total (unique IDs): {len(control_ids)}")
    print(f"   Control games missing files: {len(missing_control)}")
    if missing_control:
        for cid in sorted(missing_control):
            parent = control_to_success.get(cid, "_manual/_fill")
            print(f"      - {cid}  (control for {parent})")
    print()

    # ---- 2. Audit each game ----
    print("-" * 80)
    print("2. HISTORY COVERAGE AUDIT")
    print("-" * 80)

    success_results = []
    for sid, name in success_ids.items():
        r = audit_history_file(sid, name)
        r["group"] = "success"
        success_results.append(r)

    control_results = []
    for cid in sorted(control_ids):
        r = audit_history_file(cid)
        r["group"] = "control"
        control_results.append(r)

    all_results = success_results + control_results

    # Summary stats
    games_with_files = [r for r in all_results if r["file_exists"]]
    games_with_history = [r for r in all_results if r["history_exists"]]

    print(f"   Total games expected: {len(all_results)}")
    print(f"   Games with history files: {len(games_with_files)}")
    print(f"   Games with non-empty history: {len(games_with_history)}")
    print()

    if games_with_history:
        pre_launch_vals = [
            r["pre_launch_days"] for r in games_with_history
            if r["pre_launch_days"] is not None
        ]
        if pre_launch_vals:
            print(f"   Pre-launch coverage (games with release date):")
            print(f"      Min: {min(pre_launch_vals)} days")
            print(f"      Max: {max(pre_launch_vals)} days")
            print(f"      Median: {sorted(pre_launch_vals)[len(pre_launch_vals)//2]} days")
            print(f"      Mean: {sum(pre_launch_vals)/len(pre_launch_vals):.0f} days")
            below_30 = sum(1 for v in pre_launch_vals if v < 30)
            print(f"      Games with <30 days pre-launch: {below_30}")

        foll_pcts = [r["followers_non_null_pct"] for r in games_with_history]
        wish_pcts = [r["wishlists_non_null_pct"] for r in games_with_history]
        print()
        print(f"   Followers coverage (non-null %):")
        print(f"      Min: {min(foll_pcts)}%  Max: {max(foll_pcts)}%  Mean: {sum(foll_pcts)/len(foll_pcts):.1f}%")
        print(f"   Wishlists coverage (non-null %):")
        print(f"      Min: {min(wish_pcts)}%  Max: {max(wish_pcts)}%  Mean: {sum(wish_pcts)/len(wish_pcts):.1f}%")
    print()

    # ---- 3. Problem games ----
    print("-" * 80)
    print("3. PROBLEM GAMES")
    print("-" * 80)

    success_problems = [r for r in success_results if r["issues"]]
    control_problems = [r for r in control_results if r["issues"]]

    print()
    print(f"  === SUCCESS GAMES WITH ISSUES ({len(success_problems)}/{len(success_results)}) ===")
    if success_problems:
        for r in success_problems:
            print(f"   {r['steamId']:>10s}  {r['name']:<40s}  issues: {', '.join(r['issues'])}")
            if r["history_exists"]:
                print(f"             history: {r['history_length']} entries, "
                      f"earliest: {r['earliest_date']}, "
                      f"release: {r['release_date_str']}, "
                      f"pre-launch: {r['pre_launch_days']}d, "
                      f"followers: {r['followers_non_null_pct']}%, "
                      f"wishlists: {r['wishlists_non_null_pct']}%")
    else:
        print("   (none)")

    print()
    print(f"  === CONTROL GAMES WITH ISSUES ({len(control_problems)}/{len(control_results)}) ===")
    if control_problems:
        for r in control_problems:
            parent = control_to_success.get(r["steamId"], "_manual/_fill")
            print(f"   {r['steamId']:>10s}  (control for {parent:<10s})  issues: {', '.join(r['issues'])}")
            if r["history_exists"]:
                print(f"             history: {r['history_length']} entries, "
                      f"earliest: {r['earliest_date']}, "
                      f"release: {r['release_date_str']}, "
                      f"pre-launch: {r['pre_launch_days']}d, "
                      f"followers: {r['followers_non_null_pct']}%, "
                      f"wishlists: {r['wishlists_non_null_pct']}%")
    else:
        print("   (none)")

    # ---- 4. Detailed table: Success games ----
    print()
    print("-" * 80)
    print("4. DETAILED SUCCESS GAME COVERAGE")
    print("-" * 80)
    header = (f"{'ID':>10s}  {'Name':<35s}  {'Hist':>5s}  {'Earliest':<12s}  "
              f"{'Release':<12s}  {'PreD':>5s}  {'Foll%':>5s}  {'Wish%':>5s}  Issues")
    print(header)
    print("-" * len(header))
    for r in success_results:
        hist_len = str(r["history_length"]) if r["history_exists"] else "-"
        earliest = r["earliest_date"] or "-"
        release = r["release_date_str"] or "-"
        pre = str(r["pre_launch_days"]) if r["pre_launch_days"] is not None else "-"
        foll = f"{r['followers_non_null_pct']}" if r["history_exists"] else "-"
        wish = f"{r['wishlists_non_null_pct']}" if r["history_exists"] else "-"
        issues = ", ".join(r["issues"]) if r["issues"] else "OK"
        print(f"{r['steamId']:>10s}  {r['name']:<35s}  {hist_len:>5s}  {earliest:<12s}  "
              f"{release:<12s}  {pre:>5s}  {foll:>5s}  {wish:>5s}  {issues}")

    # ---- 5. Timeseries & Backdrop coverage ----
    print()
    print("-" * 80)
    print("5. TIMESERIES & BACKDROP COVERAGE")
    print("-" * 80)

    # Timeseries
    try:
        ts_df = pd.read_parquet(TIMESERIES_PATH)
        ts_ids = set(ts_df["steamId"].astype(str).unique())
        success_in_ts = sum(1 for sid in success_ids if sid in ts_ids)
        control_in_ts = sum(1 for cid in control_ids if cid in ts_ids)
        print(f"   Timeseries parquet: {len(ts_df)} rows, {len(ts_ids)} unique games")
        print(f"   Success games in timeseries: {success_in_ts}/{len(success_ids)}")
        print(f"   Control games in timeseries: {control_in_ts}/{len(control_ids)}")

        # Check followers/wishlists null rates in timeseries
        foll_null = ts_df["followers"].isna().sum()
        wish_null = ts_df["wishlists"].isna().sum()
        print(f"   Timeseries followers null rate: {100*foll_null/len(ts_df):.1f}%")
        print(f"   Timeseries wishlists null rate: {100*wish_null/len(ts_df):.1f}%")

        # Missing from timeseries
        missing_ts_success = [sid for sid in success_ids if sid not in ts_ids]
        missing_ts_control = [cid for cid in control_ids if cid not in ts_ids]
        if missing_ts_success:
            print(f"   Success games NOT in timeseries:")
            for sid in missing_ts_success:
                print(f"      - {sid} ({success_ids[sid]})")
        if missing_ts_control:
            print(f"   Control games NOT in timeseries: {len(missing_ts_control)}")
            for cid in sorted(missing_ts_control)[:20]:
                print(f"      - {cid}")
            if len(missing_ts_control) > 20:
                print(f"      ... and {len(missing_ts_control)-20} more")
    except Exception as e:
        print(f"   ERROR reading timeseries: {e}")

    print()

    # Backdrop
    try:
        bd_df = pd.read_parquet(BACKDROP_PATH)
        bd_ids = set(bd_df["steamId"].astype(str).unique())
        success_in_bd = sum(1 for sid in success_ids if sid in bd_ids)
        control_in_bd = sum(1 for cid in control_ids if cid in bd_ids)
        print(f"   Backdrop parquet: {len(bd_df)} rows, {len(bd_ids)} unique games")
        print(f"   Success games in backdrop: {success_in_bd}/{len(success_ids)}")
        print(f"   Control games in backdrop: {control_in_bd}/{len(control_ids)}")
    except Exception as e:
        print(f"   ERROR reading backdrop: {e}")

    # ---- 6. Overall summary ----
    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    total_expected = len(success_ids) + len(control_ids)
    total_have_files = sum(1 for r in all_results if r["file_exists"])
    total_have_history = sum(1 for r in all_results if r["history_exists"])
    total_with_issues = sum(1 for r in all_results if r["issues"])
    print(f"   Expected games:        {total_expected}")
    print(f"   Have history files:    {total_have_files} ({100*total_have_files/total_expected:.1f}%)")
    print(f"   Have non-empty history:{total_have_history} ({100*total_have_history/total_expected:.1f}%)")
    print(f"   Games with issues:     {total_with_issues} ({100*total_with_issues/total_expected:.1f}%)")
    print(f"     - Success w/ issues: {len(success_problems)}/{len(success_results)}")
    print(f"     - Control w/ issues: {len(control_problems)}/{len(control_results)}")

    issue_counts = {}
    for r in all_results:
        for iss in r["issues"]:
            tag = iss.split(" ")[0]
            issue_counts[tag] = issue_counts.get(tag, 0) + 1
    if issue_counts:
        print()
        print("   Issue breakdown:")
        for tag, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"      {tag}: {count}")

    print()
    print("Audit complete.")


if __name__ == "__main__":
    main()
