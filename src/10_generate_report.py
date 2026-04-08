"""Generate the final interactive HTML report and corresponding MD text file.

Reads all analysis data, extracts chart data, and produces:
  - output/report.html  — self-contained interactive web report (Chinese)
  - output/report_text.md — text-only version for editing/polishing
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"

ANALYSIS_FILE = DATA_DIR / "analysis_features.parquet"
FORWARD_FILE = DATA_DIR / "forward_observations.parquet"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"
HISTORIES_DIR = DATA_DIR / "histories"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"
CONTROL_FILE = CONFIG_DIR / "control_group.json"

# ── Chinese game name mapping ──────────────────────────────────────────
CN = {
    "Black Myth: Wukong": "黑神话：悟空",
    "HELLDIVERS 2": "绝地潜兵2",
    "Palworld": "幻兽帕鲁",
    "Marvel Rivals": "漫威争锋",
    "ARC Raiders": "ARC Raiders",
    "Sons Of The Forest": "森林之子",
    "Clair Obscur: Expedition 33": "光与影：远征队33",
    "Kingdom Come: Deliverance II": "天国：拯救2",
    "R.E.P.O.": "R.E.P.O.",
    "Lethal Company": "致命公司",
    "Schedule I": "Schedule I",
    "Where Winds Meet": "燕云十六声",
    "Enshrouded": "Enshrouded",
    "PEAK": "PEAK",
    "Manor Lords": "Manor Lords",
    "Balatro": "小丑牌",
    "THE FINALS": "THE FINALS",
    "Dispatch": "Dispatch",
    "Lies of P": "Lies of P",
    "Escape From Duckov": "逃离鸭科夫",
    "Mewgenics": "Mewgenics",
    "Abiotic Factor": "Abiotic Factor",
    "Gray Zone Warfare": "Gray Zone Warfare",
    "RV There Yet?": "RV There Yet?",
    "Content Warning": "Content Warning",
    "Megabonk": "Megabonk",
    "Dark and Darker": "Dark and Darker",
    "Buckshot Roulette": "恶魔轮盘",
    "Heartopia": "Heartopia",
    "YAPYAP": "YAPYAP",
    # Key control games
    "Battlefield™ 6": "战地6",
    "Ghost of Tsushima DIRECTOR'S CUT": "对马岛之魂",
    "Split Fiction": "Split Fiction",
    "S.T.A.L.K.E.R. 2: Heart of Chornobyl": "潜行者2",
    "Warhammer 40,000: Space Marine 2": "战锤40K：星际战士2",
    "Delta Force": "三角洲部队",
    "FINAL FANTASY VII REBIRTH": "最终幻想7 重生",
    "Dune: Awakening": "沙丘：觉醒",
    "Slay the Spire 2": "杀戮尖塔2",
    "Indiana Jones and the Great Circle": "印第安纳琼斯",
    "Arena Breakout: Infinite": "暗区突围：无限",
    "FragPunk": "FragPunk",
    "Mecha BREAK": "机甲战魂",
    "Avowed": "Avowed",
    "Nightingale": "Nightingale",
    "ASKA": "ASKA",
    "Dying Light: The Beast": "消逝的光芒：野兽",
    "No Rest for the Wicked": "恶人无安宁",
    "WUCHANG: Fallen Feathers": "雾常：落羽",
    "Horizon Forbidden West™ Complete Edition": "地平线 西之绝境",
    "Pacific Drive": "Pacific Drive",
    "Atomfall": "Atomfall",
    "Granblue Fantasy: Relink": "碧蓝幻想 Relink",
    "Marathon": "Marathon",
    "Grounded 2": "Grounded 2",
    "SMITE 2": "神之浩劫2",
    "Lightyear Frontier": "Lightyear Frontier",
    "PIONER": "PIONER",
    "StarRupture": "StarRupture",
    "Synergy": "Synergy",
    "Nightingale": "Nightingale",
    "Ashes of Creation": "Ashes of Creation",
    "RuneScape: Dragonwilds": "RuneScape: Dragonwilds",
    "Jump Space": "Jump Space",
    "skate.": "skate.",
}


def cn(name):
    """Get Chinese name if available, otherwise return original."""
    return CN.get(str(name), str(name))


# ── Data Loading ───────────────────────────────────────────────────────

def load_all_data():
    af = pd.read_parquet(ANALYSIS_FILE)
    fwd = pd.read_parquet(FORWARD_FILE)
    bd = pd.read_parquet(BACKDROP_FILE)
    bd["steamId"] = bd["steamId"].astype(str)
    success_list = json.loads(SUCCESS_FILE.read_text())
    success_names = {str(g["steamId"]): g["name"] for g in success_list}
    name_lookup = dict(zip(bd["steamId"], bd["name"]))
    name_lookup.update(success_names)
    return af, fwd, bd, success_names, name_lookup


# ── Data Processing ───────────────────────────────────────────────────

def compute_cohens_d(sv, cv):
    if len(sv) < 2 or len(cv) < 2:
        return 0
    pooled = ((sv.std() ** 2 + cv.std() ** 2) / 2) ** 0.5
    return abs(sv.mean() - cv.mean()) / pooled if pooled > 0 else 0


def extract_separation_power(af):
    """Extract Cohen's d for all features."""
    features = [
        ("follower_velocity_30d", "关注者增速 - 上线前30天", True),
        ("follower_velocity_90d", "关注者增速 - 上线前90天", True),
        ("follower_velocity_180d", "关注者增速 - 上线前180天", True),
        ("followers_at_launch", "上线时关注者数", True),
        ("wishlist_velocity_180d", "愿望单增速 - 上线前180天", False),
        ("wishlist_velocity_90d", "愿望单增速 - 上线前90天", False),
        ("wishlist_velocity_30d", "愿望单增速 - 上线前30天", False),
        ("wishlists_at_launch", "上线时愿望单数", False),
        ("prior_game_count", "开发者历史作品数", None),
        ("tag_count", "标签数", None),
        ("price_at_launch", "售价", None),
        ("days_on_steam_pre_launch", "上线前挂页天数", None),
        ("prior_avg_revenue", "开发者历史平均营收", None),
    ]
    results = []
    for col, label, is_follower in features:
        if col not in af.columns:
            continue
        sv = af[af["group"] == "success"][col].dropna()
        cv = af[af["group"] == "control"][col].dropna()
        d = compute_cohens_d(sv, cv)
        results.append({
            "col": col, "label": label, "d": round(d, 2),
            "is_follower": is_follower,
            "n_s": len(sv), "n_c": len(cv),
            "s_median": round(sv.median(), 1) if not sv.empty else 0,
            "c_median": round(cv.median(), 1) if not cv.empty else 0,
        })
    # Sort by category: follower first, wishlist second, other third; within each descending by d
    def _group_order(x):
        if x["is_follower"] is True:
            return (0, -x["d"])
        elif x["is_follower"] is False:
            return (1, -x["d"])
        else:
            return (2, -x["d"])
    results.sort(key=_group_order)
    return results


def extract_success_games_table(af, bd, success_names):
    """Build success games table data."""
    games = []
    for sid, name in success_names.items():
        row = af[af["steamId"] == sid]
        bd_row = bd[bd["steamId"] == sid]
        copies = 0
        launch = ""
        pub = ""
        if len(bd_row) > 0:
            c = bd_row.iloc[0].get("copiesSold", 0)
            if pd.notna(c):
                copies = int(c)
            pubs = bd_row.iloc[0].get("publishers", [])
            if hasattr(pubs, '__iter__') and not isinstance(pubs, str):
                pub = ", ".join(str(p) for p in pubs)
            else:
                pub = str(pubs)
        if len(row) > 0:
            r = row.iloc[0]
            ld = r.get("launch_date", None)
            if pd.notna(ld):
                launch = str(ld)[:10]
        games.append({
            "name": cn(name), "en_name": name,
            "copies_sold": copies, "launch": launch,
            "publisher": pub, "steamId": sid,
        })
    games.sort(key=lambda x: x["copies_sold"], reverse=True)
    return games


def extract_trajectories(fwd, game_ids, name_lookup):
    """Extract follower trajectory data for specific games."""
    fl = fwd[(fwd["metric"] == "followers")].copy()
    trajectories = {}
    for sid in game_ids:
        g = fl[fl["steamId"] == sid].sort_values("days_to_launch", ascending=False)
        g = g[(g["days_to_launch"] >= -7) & (g["days_to_launch"] <= 600)]
        if len(g) < 3:
            continue
        name = cn(name_lookup.get(sid, sid))
        trajectories[name] = [
            [int(r["days_to_launch"]), float(r["total"])]
            for _, r in g.iterrows()
            if pd.notna(r["total"])
        ]
    return trajectories


def extract_control_envelope(fwd):
    """Compute median + P25/P75 follower trajectory for control group."""
    fl = fwd[(fwd["metric"] == "followers") & (fwd["group"] == "control")].copy()
    fl = fl[(fl["days_to_launch"] >= -30) & (fl["days_to_launch"] <= 600)]
    grouped = fl.groupby("days_to_launch")["total"].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    grouped.columns = ["median", "p25", "p75"]
    grouped = grouped.reset_index().sort_values("days_to_launch", ascending=False)
    return [[int(r["days_to_launch"]), float(r["median"])] for _, r in grouped.iterrows()]


def extract_peak_week_data(fwd, name_lookup):
    """Extract peak week velocity per game from forward_observations (success + control only).
    Used for charts that need the original success/control grouping."""
    fl = fwd[(fwd["metric"] == "followers") & (fwd["days_to_launch"] > 0)].copy()
    games = []
    for sid, g in fl.groupby("steamId"):
        group = g["group"].iloc[0]
        has_onset = g["onset_artifact"].iloc[0] if "onset_artifact" in g.columns else False
        rates = g["daily_growth_7d"].dropna()
        if has_onset and len(rates) > 1:
            rates = rates.iloc[1:]
        if rates.empty:
            continue
        name = cn(name_lookup.get(sid, sid))
        peak = rates.max()
        avg = rates.median()
        # Find the date of peak
        peak_idx = rates.idxmax()
        peak_dtl = g.loc[peak_idx, "days_to_launch"] if peak_idx in g.index else None
        games.append({
            "steamId": sid, "name": name, "group": group,
            "peak_week": round(float(peak), 1),
            "avg_daily": round(float(avg), 1),
            "peak_dtl": int(peak_dtl) if pd.notna(peak_dtl) else None,
        })
    return games


def extract_peak_week_from_histories(success_ids, name_lookup):
    """Compute peak 7-day follower velocity from ALL history JSON files.

    Returns a list of dicts with steamId, name, group (success/backdrop),
    peak_week, avg_daily, peak_dtl for every game with sufficient history.
    Used for strategy evaluation against the full backdrop population.
    """
    games = []
    history_files = sorted(HISTORIES_DIR.glob("*.json"))
    skipped = 0

    for hf in history_files:
        sid = hf.stem
        try:
            data = json.loads(hf.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        history = data.get("history", [])
        if len(history) < 3:
            skipped += 1
            continue

        # Get release date
        release_ms = data.get("releaseDate") or data.get("firstReleaseDate")
        if not release_ms:
            skipped += 1
            continue

        release_date = pd.Timestamp(release_ms, unit="ms")

        # Build follower timeseries
        rows = []
        for entry in history:
            ts = entry.get("timeStamp") or entry.get("date")
            fol = entry.get("followers")
            if ts is not None and fol is not None:
                rows.append({"date": pd.Timestamp(ts, unit="ms"), "followers": int(fol)})

        if len(rows) < 3:
            skipped += 1
            continue

        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
        df["days_to_launch"] = (release_date - df["date"]).dt.days

        # Pre-launch only (days_to_launch > 0)
        pre = df[df["days_to_launch"] > 0].copy()
        if len(pre) < 2:
            skipped += 1
            continue

        # Compute daily growth rate
        pre = pre.sort_values("date")
        pre["fol_diff"] = pre["followers"].diff()
        pre["date_diff"] = pre["date"].diff().dt.total_seconds() / 86400
        pre["daily_rate"] = pre["fol_diff"] / pre["date_diff"]

        # Onset artifact: if first entry has large followers, null out first rate
        if pre.iloc[0]["followers"] > 100 and len(pre) > 1:
            pre.iloc[0, pre.columns.get_loc("daily_rate")] = np.nan

        # 7-day rolling mean
        rates = pre["daily_rate"].dropna()
        if len(rates) < 2:
            skipped += 1
            continue

        # Simple rolling window (by position, since entries may not be daily)
        if len(rates) >= 7:
            rolling = rates.rolling(7, min_periods=4).mean().dropna()
        else:
            rolling = rates

        if rolling.empty:
            skipped += 1
            continue

        peak = float(rolling.max())
        avg = float(rolling.median())
        peak_idx = rolling.idxmax()
        peak_dtl = int(pre.loc[peak_idx, "days_to_launch"]) if peak_idx in pre.index else None

        group = "success" if sid in success_ids else "backdrop"
        name = cn(name_lookup.get(sid, data.get("name", sid)))
        revenue = data.get("revenue") or data.get("totalRevenue") or 0

        games.append({
            "steamId": sid, "name": name, "group": group,
            "peak_week": round(peak, 1),
            "avg_daily": round(avg, 1),
            "peak_dtl": peak_dtl,
            "revenue": int(revenue) if revenue else 0,
        })

    print(f"  Computed peak velocity for {len(games)} games from {len(history_files)} history files (skipped {skipped})")
    return games


def extract_threshold_sweep(peak_data, avg_floor=0):
    """Sweep thresholds on peak_week velocity with optional avg_daily floor."""
    gf = pd.DataFrame(peak_data)
    if avg_floor > 0:
        gf = gf[gf["avg_daily"] >= avg_floor]
    n_success = (gf["group"] == "success").sum()
    base_pool = gf.copy()
    values = gf["peak_week"].values
    thresholds = sorted(set(
        list(np.percentile(values, np.arange(5, 100, 2.5))) +
        [50, 100, 200, 300, 400, 500, 600, 750, 1000, 1100, 1250, 1500, 1900, 2000, 3000, 5000]
    ))
    rows = []
    for t in thresholds:
        flagged = base_pool[base_pool["peak_week"] >= t]
        n_f = len(flagged)
        if n_f == 0:
            continue
        tp = (flagged["group"] == "success").sum()
        prec = tp / n_f
        rec = tp / n_success if n_success > 0 else 0
        rows.append({
            "threshold": round(float(t), 0),
            "flagged": int(n_f), "hits": int(tp),
            "precision": round(float(prec), 3), "recall": round(float(rec), 3),
        })
    return rows


def extract_strategy_tiers(peak_data, name_lookup, success_names):
    """Build 4-tier strategy data with dual thresholds (peak + avg)."""
    gf = pd.DataFrame(peak_data)
    evaluable_success = set(gf[gf["group"] == "success"]["steamId"])
    all_success_ids = set(success_names.keys())
    not_evaluable = all_success_ids - evaluable_success

    # Dual thresholds: (cn_label, en_label, peak_threshold, avg_threshold, color)
    tiers = [
        ("广角策略", "Wide Angle", 400, 20, "#10b981"),
        ("精选策略", "Curated", 900, 40, "#3b82f6"),
        ("高确信策略", "High Conviction", 1900, 70, "#ef4444"),
    ]
    result = []
    for cn_label, en_label, peak_t, avg_t, color in tiers:
        flagged = gf[
            (gf["peak_week"] >= peak_t) & (gf["avg_daily"] >= avg_t)
        ].sort_values("peak_week", ascending=False)
        hits = flagged[flagged["group"] == "success"]
        fps = flagged[flagged["group"] != "success"]
        missed = gf[
            (gf["group"] == "success") &
            ~((gf["peak_week"] >= peak_t) & (gf["avg_daily"] >= avg_t))
        ]
        not_eval_names = sorted([cn(success_names.get(sid, sid)) for sid in not_evaluable])

        # Compute lift over base rate
        base_rate = len(evaluable_success) / len(gf) if len(gf) > 0 else 0
        prec = len(hits) / len(flagged) if len(flagged) > 0 else 0
        lift = round(prec / base_rate, 0) if base_rate > 0 else 0

        result.append({
            "cn_label": cn_label, "en_label": en_label,
            "peak_threshold": peak_t, "avg_threshold": avg_t,
            "threshold": peak_t,  # backward compat
            "color": color,
            "radar": len(flagged),
            "hit_count": len(hits), "fp_count": len(fps),
            "miss_count": len(missed), "no_data_count": len(not_evaluable),
            "evaluable_total": len(evaluable_success),
            "precision": round(prec, 3),
            "recall": round(len(hits) / len(evaluable_success), 3) if len(evaluable_success) > 0 else 0,
            "lift": int(lift),
            "hits": [{"name": r["name"], "peak": round(r["peak_week"], 0), "avg": round(r["avg_daily"], 0)}
                     for _, r in hits.iterrows()],
            "misses": [{"name": r["name"], "peak": round(r["peak_week"], 0), "avg": round(r["avg_daily"], 0)}
                       for _, r in missed.iterrows()],
            "fps": [{"name": r["name"], "peak": round(r["peak_week"], 0), "avg": round(r["avg_daily"], 0)}
                    for _, r in fps.iterrows()],
            "no_data": not_eval_names,
            "watchlist": [{"name": r["name"], "peak": round(r["peak_week"], 0),
                           "avg": round(r["avg_daily"], 0), "group": r["group"]}
                          for _, r in flagged.iterrows()],
        })
    return result


def extract_strategy_quality(peak_data, bd=None):
    """Compute revenue-based quality metrics for each strategy tier.
    If bd (full backdrop DataFrame) is provided, Top100 is computed against the full population."""
    gf = pd.DataFrame(peak_data)
    # Top100 threshold: use full backdrop if available, else fall back to peak_data pool
    if bd is not None:
        full_rev = bd[bd["revenue"] > 0]["revenue"].sort_values(ascending=False)
        top100_threshold = float(full_rev.iloc[99]) if len(full_rev) >= 100 else 0
        total_pool = len(bd)
    else:
        full_rev = gf[gf["revenue"] > 0]["revenue"].sort_values(ascending=False)
        top100_threshold = float(full_rev.iloc[99]) if len(full_rev) >= 100 else 0
        total_pool = len(gf)
    all_rev = gf[gf["revenue"] > 0]["revenue"].sort_values(ascending=False)
    rand_median = float(all_rev.median()) if len(all_rev) > 0 else 1

    tiers = [
        ("广角策略", 400, 20),
        ("精选策略", 900, 40),
        ("高确信策略", 1900, 70),
    ]
    # $50M+ hit/recall stats
    total_50m = int((gf["revenue"] > 50e6).sum())

    results = []
    for label, pt, at in tiers:
        flagged = gf[(gf["peak_week"] >= pt) & (gf["avg_daily"] >= at)]
        n = len(flagged)
        f_rev = flagged[flagged["revenue"] > 0]["revenue"]
        n_10m = int((f_rev > 10e6).sum())
        n_50m = int((f_rev > 50e6).sum())
        med = float(f_rev.median()) if len(f_rev) > 0 else 0
        n_top100 = int((f_rev >= top100_threshold).sum())
        fps_50m = n - n_50m

        results.append({
            "label": label, "peak_t": pt, "avg_t": at, "n": n,
            "pct_10m": round(n_10m / n, 3) if n > 0 else 0,
            "pct_50m": round(n_50m / n, 3) if n > 0 else 0,
            "n_10m": n_10m, "n_50m": n_50m,
            "median_rev": round(med / 1e6, 1),
            "pct_top100": round(n_top100 / n, 3) if n > 0 else 0,
            "n_top100": n_top100,
            "games_per_10m": round(n / n_10m, 1) if n_10m > 0 else 0,
            "rev_lift": round(med / rand_median, 0) if rand_median > 0 else 0,
            "hit_50m_precision": round(n_50m / n, 3) if n > 0 else 0,
            "hit_50m_recall": round(n_50m / total_50m, 3) if total_50m > 0 else 0,
            "fps_50m": fps_50m,
        })

    return {
        "tiers": results,
        "rand_median": round(rand_median / 1e6, 2),
        "top100_threshold": round(top100_threshold / 1e6, 1),
        "total_games": total_pool,
        "total_50m": total_50m,
    }


def extract_tag_analysis(af):
    """Extract tag over/under-representation."""
    from collections import Counter
    s_tags = Counter()
    c_tags = Counter()
    n_s = (af["group"] == "success").sum()
    n_c = (af["group"] == "control").sum()

    for _, row in af.iterrows():
        tags = row.get("tags", [])
        if not hasattr(tags, '__iter__') or isinstance(tags, str):
            continue
        counter = s_tags if row["group"] == "success" else c_tags
        for t in tags:
            counter[str(t)] += 1

    diffs = []
    all_tags = set(s_tags.keys()) | set(c_tags.keys())
    for tag in all_tags:
        s_pct = s_tags[tag] / n_s * 100 if n_s > 0 else 0
        c_pct = c_tags[tag] / n_c * 100 if n_c > 0 else 0
        diff = s_pct - c_pct
        if abs(diff) > 5:
            diffs.append({"tag": tag, "diff": round(diff, 1), "s_pct": round(s_pct, 1), "c_pct": round(c_pct, 1)})
    diffs.sort(key=lambda x: x["diff"], reverse=True)
    return diffs


def extract_velocity_windows(fwd):
    """Extract velocity by time window."""
    fl = fwd[fwd["metric"] == "followers"].copy()
    windows = [
        ("T-360→T-180", -360, -180),
        ("T-180→T-90", -180, -90),
        ("T-90→T-30", -90, -30),
        ("T-30→上线", -30, 0),
    ]
    results = []
    for label, start, end in windows:
        w = fl[(fl["days_to_launch"] <= -start) & (fl["days_to_launch"] > -end)]
        for group in ["success", "control"]:
            gw = w[w["group"] == group]
            vels = []
            for sid, sg in gw.groupby("steamId"):
                if len(sg) < 5:
                    continue
                rates = sg["daily_growth_7d"].dropna()
                if rates.empty:
                    total_s = sg["total"].iloc[0]
                    total_e = sg["total"].iloc[-1]
                    days = abs((sg["date"].iloc[-1] - sg["date"].iloc[0]).days)
                    if days > 0 and pd.notna(total_s) and pd.notna(total_e):
                        vels.append((total_e - total_s) / days)
                else:
                    vels.append(rates.median())
            if vels:
                results.append({
                    "window": label, "group": group,
                    "median": round(float(np.median(vels)), 1),
                    "n": len(vels),
                })
    return results


def extract_scatter_data(af, name_lookup, peak_data_full=None):
    """Extract followers-at-launch vs revenue data.

    When peak_data_full is provided, builds scatter from the full backdrop
    sample using last pre-launch follower count from history JSON files.
    """
    if peak_data_full:
        # Build from full backdrop: use history files for follower-at-launch
        points = []
        for game in peak_data_full:
            sid = game["steamId"]
            rev = game.get("revenue", 0)
            if not rev or rev <= 0:
                continue
            # Get followers at launch from history file
            hf = HISTORIES_DIR / f"{sid}.json"
            if not hf.exists():
                continue
            try:
                data = json.loads(hf.read_text(encoding="utf-8"))
            except Exception:
                continue
            release_ms = data.get("releaseDate") or data.get("firstReleaseDate")
            if not release_ms:
                continue
            # Find last pre-launch follower count
            history = data.get("history", [])
            release_date = pd.Timestamp(release_ms, unit="ms")
            fol_at_launch = 0
            for entry in history:
                ts = entry.get("timeStamp") or entry.get("date")
                fol = entry.get("followers")
                if ts and fol:
                    dt = pd.Timestamp(ts, unit="ms")
                    if dt <= release_date:
                        fol_at_launch = int(fol)
            if fol_at_launch <= 0:
                continue
            points.append({
                "name": game["name"],
                "followers": fol_at_launch,
                "revenue": round(float(rev) / 1e6, 1),
                "group": game["group"],
            })
        return points

    # Fallback: original analysis_features approach
    cols = ["steamId", "followers_at_launch", "first_month_revenue", "group"]
    df = af[cols].dropna()
    df = df[(df["followers_at_launch"] > 0) & (df["first_month_revenue"] > 0)]
    points = []
    for _, r in df.iterrows():
        name = cn(name_lookup.get(str(r["steamId"]), str(r["steamId"])))
        points.append({
            "name": name,
            "followers": int(r["followers_at_launch"]),
            "revenue": round(float(r["first_month_revenue"]) / 1e6, 1),
            "group": r["group"],
        })
    return points


def extract_trajectory_with_peak(fwd, game_ids, name_lookup, peak_data):
    """Extract trajectory + peak annotation for select games."""
    fl = fwd[(fwd["metric"] == "followers")].copy()
    peak_lookup = {g["steamId"]: g for g in peak_data}
    trajectories = []
    for sid in game_ids:
        g = fl[(fl["steamId"] == sid) & (fl["days_to_launch"] > 0)]
        g = g.sort_values("days_to_launch", ascending=False)
        if len(g) < 5:
            continue
        name = cn(name_lookup.get(sid, sid))
        data_points = [[int(r["days_to_launch"]), round(float(r["daily_growth_7d"]), 1)]
                        for _, r in g.iterrows() if pd.notna(r["daily_growth_7d"])]
        peak_info = peak_lookup.get(sid, {})
        trajectories.append({
            "name": name, "steamId": sid,
            "data": data_points,
            "peak_week": peak_info.get("peak_week", 0),
            "peak_dtl": peak_info.get("peak_dtl"),
        })
    return trajectories


def extract_detection_leadtime(fwd, success_names):
    """For each strategy threshold, count how many success games are detected
    ≥90 days before launch, <90 days, missed, or have no data."""
    fl = fwd[
        (fwd["group"] == "success") &
        (fwd["metric"] == "followers") &
        (fwd["days_to_launch"] >= 0) &
        (fwd["daily_growth_7d"].notna())
    ].copy()

    thresholds = [
        ("广角策略", 400),
        ("精选策略", 900),
        ("高确信策略", 1900),
    ]
    results = []
    for label, thresh in thresholds:
        early, late, missed, nodata = 0, 0, 0, 0
        early_names, late_names = [], []
        for sid, name in success_names.items():
            gd = fl[fl["steamId"] == sid]
            if len(gd) == 0:
                nodata += 1
                continue
            above = gd[gd["daily_growth_7d"] >= thresh]
            if len(above) > 0:
                first_day = int(above["days_to_launch"].max())
                if first_day >= 90:
                    early += 1
                    early_names.append(cn(name))
                else:
                    late += 1
                    late_names.append(cn(name))
            else:
                missed += 1
        results.append({
            "label": label, "threshold": thresh,
            "early": early, "late": late, "missed": missed, "nodata": nodata,
            "early_names": early_names, "late_names": late_names,
        })
    return results


def extract_detection_leadtime_50m(peak_data_full, name_lookup=None, min_n=30):
    """For each $50M+ game, compute detection lead time (T- days) under each strategy.
    Uses real-time simulation: cumulative peak and median at each rolling point.
    Requires at least min_n rolling observations before avg_daily condition can trigger."""
    gf = pd.DataFrame(peak_data_full)
    name_lookup = name_lookup or {}
    big_ids = set(gf[gf["revenue"] > 50e6]["steamId"].tolist())
    total_50m = len(big_ids)

    tiers = [
        ("广角策略", 400, 20, "#10b981"),
        ("精选策略", 900, 40, "#3b82f6"),
        ("高确信策略", 1900, 70, "#ef4444"),
    ]

    # For each $50M+ game, simulate detection
    per_game = []  # list of {steamId, name, revenue, tier_label: T-days or None}
    for hf in sorted(HISTORIES_DIR.glob("*.json")):
        sid = hf.stem
        if sid not in big_ids:
            continue
        data = json.loads(hf.read_text(encoding="utf-8"))
        history = data.get("history", [])
        if len(history) < 3:
            continue
        release_ms = data.get("releaseDate") or data.get("firstReleaseDate")
        if not release_ms:
            continue
        release_date = pd.Timestamp(release_ms, unit="ms")

        rows = []
        for entry in history:
            ts = entry.get("timeStamp") or entry.get("date")
            fol = entry.get("followers")
            if ts is not None and fol is not None:
                rows.append({"date": pd.Timestamp(ts, unit="ms"), "followers": int(fol)})
        if len(rows) < 3:
            continue

        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
        df["days_to_launch"] = (release_date - df["date"]).dt.days
        pre = df[df["days_to_launch"] > 0].copy().sort_values("date")
        if len(pre) < 2:
            continue
        pre["fol_diff"] = pre["followers"].diff()
        pre["date_diff"] = pre["date"].diff().dt.total_seconds() / 86400
        pre["daily_rate"] = pre["fol_diff"] / pre["date_diff"]
        if pre.iloc[0]["followers"] > 100 and len(pre) > 1:
            pre.iloc[0, pre.columns.get_loc("daily_rate")] = np.nan
        rates = pre["daily_rate"].dropna()
        if len(rates) < 2:
            continue
        if len(rates) >= 7:
            rolling = rates.rolling(7, min_periods=4).mean().dropna()
        else:
            rolling = rates
        if rolling.empty:
            continue

        detection = {}
        rolling_values = []
        for idx, val in rolling.items():
            rolling_values.append(val)
            n = len(rolling_values)
            cur_peak = max(rolling_values)
            cur_avg = float(np.median(rolling_values))
            dtl = int(pre.loc[idx, "days_to_launch"])
            for label, pt, at, _ in tiers:
                if label not in detection and cur_peak >= pt and n >= min_n and cur_avg >= at:
                    detection[label] = dtl

        revenue = data.get("revenue") or data.get("totalRevenue") or 0
        name = cn(name_lookup.get(sid, data.get("name", sid)))
        rec = {"steamId": sid, "name": name, "revenue": int(revenue)}
        for label, _, _, _ in tiers:
            rec[label] = detection.get(label)
        per_game.append(rec)

    # Find $50M+ games with insufficient pre-launch data (< min_n rolling points)
    # These are games that CANNOT be evaluated under the min_n requirement
    short_data_names = set()
    for hf in sorted(HISTORIES_DIR.glob("*.json")):
        sid = hf.stem
        if sid not in big_ids:
            continue
        data = json.loads(hf.read_text(encoding="utf-8"))
        history = data.get("history", [])
        if len(history) < 3:
            continue
        release_ms = data.get("releaseDate") or data.get("firstReleaseDate")
        if not release_ms:
            continue
        release_date = pd.Timestamp(release_ms, unit="ms")
        rows = []
        for entry in history:
            ts = entry.get("timeStamp") or entry.get("date")
            fol = entry.get("followers")
            if ts is not None and fol is not None:
                rows.append({"date": pd.Timestamp(ts, unit="ms"), "followers": int(fol)})
        if len(rows) < 3:
            continue
        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
        df["days_to_launch"] = (release_date - df["date"]).dt.days
        pre = df[df["days_to_launch"] > 0]
        if len(pre) < min_n:
            gname = name_lookup.get(sid, data.get("name", sid))
            short_data_names.add(cn(gname) if gname else sid)
    short_data_games = [{"name": n} for n in sorted(short_data_names)]

    # Aggregate stats per tier
    pgf = pd.DataFrame(per_game)
    tier_stats = []
    for label, pt, at, color in tiers:
        mask = pgf[label].notna()
        vals = pgf.loc[mask, label].values
        names = pgf.loc[mask, "name"].values
        detected = len(vals)
        missed = total_50m - detected
        # Sort by days and keep names paired
        paired = sorted(zip(vals.tolist(), names.tolist()), key=lambda x: x[0])
        tier_stats.append({
            "label": label, "color": color,
            "detected": detected, "missed": missed, "total": total_50m,
            "values": [p[0] for p in paired],
            "names": [p[1] for p in paired],
            "median": float(np.median(vals)) if len(vals) > 0 else 0,
            "mean": float(np.mean(vals)) if len(vals) > 0 else 0,
            "q1": float(np.percentile(vals, 25)) if len(vals) > 0 else 0,
            "q3": float(np.percentile(vals, 75)) if len(vals) > 0 else 0,
            "min": float(np.min(vals)) if len(vals) > 0 else 0,
            "max": float(np.max(vals)) if len(vals) > 0 else 0,
        })

    return {"tiers": tier_stats, "total_50m": total_50m, "per_game": per_game,
            "min_n": min_n, "short_data_games": short_data_games}


def extract_peak_cohens_d(peak_data, peak_data_full=None):
    """Compute Cohen's d for peak_week and avg_daily.

    Uses log-transformed values on full backdrop for more meaningful comparison.
    Falls back to control group if peak_data_full not provided.
    """
    # On full backdrop (log scale) — primary for v4
    if peak_data_full:
        gf = pd.DataFrame(peak_data_full)
        sv_peak = np.log10(gf[gf["group"] == "success"]["peak_week"].clip(lower=1))
        bv_peak = np.log10(gf[gf["group"] != "success"]["peak_week"].clip(lower=1))
        sv_avg = np.log10(gf[gf["group"] == "success"]["avg_daily"].clip(lower=1))
        bv_avg = np.log10(gf[gf["group"] != "success"]["avg_daily"].clip(lower=1))
        d_peak_log = compute_cohens_d(sv_peak, bv_peak)
        d_avg_log = compute_cohens_d(sv_avg, bv_avg)

        # Also raw medians for display
        s = gf[gf["group"] == "success"]
        b = gf[gf["group"] != "success"]
        return {
            "peak_week_d": round(d_peak_log, 2),
            "avg_daily_d": round(d_avg_log, 2),
            "peak_s_median": round(float(s["peak_week"].median()), 0),
            "peak_b_median": round(float(b["peak_week"].median()), 0),
            "avg_s_median": round(float(s["avg_daily"].median()), 0),
            "avg_b_median": round(float(b["avg_daily"].median()), 0),
            "peak_ratio": round(float(s["peak_week"].median()) / max(float(b["peak_week"].median()), 0.1), 1),
            "avg_ratio": round(float(s["avg_daily"].median()) / max(float(b["avg_daily"].median()), 0.1), 1),
            "n_success": int((gf["group"] == "success").sum()),
            "n_backdrop": int((gf["group"] != "success").sum()),
        }

    # Fallback: control group (raw scale)
    gf = pd.DataFrame(peak_data)
    sv_peak = gf[gf["group"] == "success"]["peak_week"].dropna()
    cv_peak = gf[gf["group"] == "control"]["peak_week"].dropna()
    sv_avg = gf[gf["group"] == "success"]["avg_daily"].dropna()
    cv_avg = gf[gf["group"] == "control"]["avg_daily"].dropna()
    d_peak = compute_cohens_d(sv_peak, cv_peak)
    d_avg = compute_cohens_d(sv_avg, cv_avg)
    return {"peak_week_d": round(d_peak, 2), "avg_daily_d": round(d_avg, 2)}


# ── HTML Generation ───────────────────────────────────────────────────

def generate_html(report_data):
    d = report_data
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>慧眼识珠：Steam爆款循迹</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root {{
  --bg: #f0f4f8; --card: #ffffff; --text: #1e293b; --heading: #0f172a;
  --primary: #1b2838; --primary-light: #2a475e; --primary-lighter: #c7d5e0;
  --green: #10b981; --green-light: #d1fae5; --blue: #3b82f6; --blue-light: #dbeafe;
  --gray: #94a3b8; --gray-light: #f1f5f9; --amber: #f59e0b; --red: #ef4444;
  --border: #e2e8f0; --radius: 12px; --shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei",
    "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.8;
  font-size: 16px; -webkit-font-smoothing: antialiased;
}}
.container {{ max-width: 900px; margin: 0 auto; padding: 0 20px; }}
/* ── Hero ── */
.hero {{
  background: linear-gradient(135deg, #1b2838 0%, #2a475e 40%, #171a21 100%);
  color: white; padding: 80px 20px 60px; text-align: center;
}}
.hero h1 {{ font-size: 2.4em; font-weight: 800; margin-bottom: 12px; letter-spacing: -0.02em; }}
.hero .subtitle {{ font-size: 1.1em; opacity: 0.85; max-width: 600px; margin: 0 auto; }}
.hero .meta {{ margin-top: 20px; font-size: 0.85em; opacity: 0.6; }}
/* ── Nav ── */
nav {{
  background: var(--card); border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 100; box-shadow: var(--shadow);
}}
nav .container {{ display: flex; gap: 0; overflow-x: auto; padding: 0; }}
nav a {{
  display: block; padding: 14px 18px; text-decoration: none; color: var(--gray);
  font-size: 0.9em; white-space: nowrap; border-bottom: 2px solid transparent;
  transition: all 0.2s;
}}
nav a:hover, nav a.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
/* ── Sections ── */
section {{ padding: 50px 0; }}
section:nth-child(even) {{ background: var(--card); }}
.section-num {{
  display: inline-block; background: var(--primary); color: white;
  width: 32px; height: 32px; border-radius: 50%; text-align: center;
  line-height: 32px; font-weight: 700; font-size: 0.9em; margin-right: 10px;
}}
h2 {{
  font-size: 1.8em; font-weight: 700; color: var(--heading);
  margin-bottom: 24px; line-height: 1.3;
}}
h3 {{
  font-size: 1.3em; font-weight: 600; color: var(--heading);
  margin: 36px 0 16px; line-height: 1.4;
}}
p {{ margin-bottom: 16px; }}
.lead {{ font-size: 1.1em; color: #475569; margin-bottom: 24px; }}
.highlight {{
  background: #e8edf2; border-left: 4px solid var(--primary);
  padding: 16px 20px; border-radius: 0 var(--radius) var(--radius) 0;
  margin: 20px 0;
}}
.highlight-blue {{
  background: var(--blue-light); border-left: 4px solid var(--blue);
  padding: 16px 20px; border-radius: 0 var(--radius) var(--radius) 0;
  margin: 20px 0;
}}
.callout {{
  background: #fffbeb; border: 1px solid #fbbf24; border-radius: var(--radius);
  padding: 16px 20px; margin: 20px 0;
}}
/* ── Charts ── */
.chart-container {{
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; margin: 24px 0; box-shadow: var(--shadow);
}}
.chart-box {{ width: 100%; height: 420px; }}
.chart-box-tall {{ width: 100%; height: 520px; }}
.chart-box-short {{ width: 100%; height: 320px; }}
.chart-caption {{
  text-align: center; font-size: 0.85em; color: var(--gray);
  margin-top: 8px; font-style: italic;
}}
/* ── Tables ── */
.data-table {{
  width: 100%; border-collapse: collapse; font-size: 0.9em;
  margin: 20px 0; background: var(--card); border-radius: var(--radius);
  overflow: hidden;
}}
.data-table th {{
  background: var(--gray-light); color: var(--heading); font-weight: 600;
  padding: 12px 16px; text-align: left; border-bottom: 2px solid var(--border);
}}
.data-table td {{
  padding: 10px 16px; border-bottom: 1px solid var(--border);
}}
.data-table tr:hover {{ background: #f8fafc; }}
.data-table .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.tag-success {{ color: var(--green); font-weight: 600; }}
.tag-control {{ color: var(--gray); }}
/* ── Strategy Tabs ── */
.strategy-tabs {{
  display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap;
}}
.strategy-tabs button {{
  padding: 10px 20px; border: 2px solid var(--border); border-radius: 999px;
  background: var(--card); cursor: pointer; font-size: 0.9em; font-weight: 600;
  transition: all 0.2s; color: var(--text);
}}
.strategy-tabs button.active {{
  border-color: var(--primary); background: var(--primary-lighter); color: var(--primary);
}}
.strategy-tabs button:hover {{ border-color: var(--primary-light); }}
.strategy-panel {{
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden;
}}
.strategy-header {{
  padding: 20px; background: var(--gray-light);
  display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 16px;
}}
.stat-card {{
  text-align: center;
}}
.stat-card .stat-value {{ font-size: 1.6em; font-weight: 700; color: var(--heading); }}
.stat-card .stat-label {{ font-size: 0.8em; color: var(--gray); }}
.strategy-columns {{
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 0;
  border-top: 1px solid var(--border);
}}
.strategy-col {{
  border-right: 1px solid var(--border); padding: 0;
}}
.strategy-col:last-child {{ border-right: none; }}
.strategy-col-header {{
  padding: 10px 12px; font-weight: 600; font-size: 0.85em;
  text-align: center; color: white;
}}
.strategy-col-body {{
  padding: 8px; max-height: 350px; overflow-y: auto; font-size: 0.8em;
}}
.strategy-col-body .game-item {{
  padding: 4px 6px; border-radius: 4px; margin-bottom: 2px;
  display: flex; justify-content: space-between; align-items: center;
}}
.strategy-col-body .game-item:hover {{ background: var(--gray-light); }}
.game-vel {{ color: var(--gray); font-size: 0.85em; font-variant-numeric: tabular-nums; }}
.divider {{ padding: 6px; text-align: center; color: var(--gray); font-size: 0.75em; border-top: 1px dashed var(--border); }}
/* ── 2x2 Grid ── */
.grid-2x2 {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 0;
  border: 2px solid var(--border); border-radius: var(--radius);
  overflow: hidden; margin: 24px 0;
}}
.grid-2x2 .cell {{
  padding: 20px; border: 1px solid var(--border);
}}
.grid-2x2 .cell h4 {{ font-size: 1em; margin-bottom: 8px; }}
.grid-2x2 .cell p {{ font-size: 0.9em; margin: 0; }}
/* ── Footer ── */
footer {{
  background: var(--heading); color: #94a3b8; padding: 40px 20px;
  text-align: center; font-size: 0.85em;
}}
/* ── Responsive ── */
@media (max-width: 768px) {{
  .hero h1 {{ font-size: 1.6em; }}
  .hero {{ padding: 50px 16px 40px; }}
  h2 {{ font-size: 1.4em; }}
  h3 {{ font-size: 1.1em; }}
  nav a {{ padding: 12px 14px; font-size: 0.8em; }}
  .strategy-columns {{ grid-template-columns: 1fr 1fr; }}
  .strategy-col {{ border-bottom: 1px solid var(--border); }}
  .chart-box {{ height: 320px; }}
  .chart-box-tall {{ height: 400px; }}
  .grid-2x2 {{ grid-template-columns: 1fr; }}
  .strategy-header {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 600px) {{
  .chart-container {{ padding: 12px; }}
  .chart-box {{ height: 280px; }}
  .chart-box-tall {{ height: 350px; }}
  section {{ padding: 30px 0; }}
  .container {{ padding: 0 12px; }}
  .strategy-col-body {{ max-height: 250px; }}
  .stat-card .stat-value {{ font-size: 1.3em; }}
  .highlight, .highlight-blue {{ padding: 12px 14px; }}
  .callout {{ padding: 12px 14px; }}
}}
@media (max-width: 480px) {{
  .strategy-columns {{ grid-template-columns: 1fr; }}
  body {{ font-size: 15px; }}
}}
/* scroll behavior */
html {{ scroll-behavior: smooth; }}
</style>
</head>
<body>

<!-- ════════ HERO ════════ -->
<header class="hero">
  <h1>「慧眼识珠」：Steam爆款循迹</h1>
  <p class="subtitle">从 Steam 易得数据出发，寻找下一个爆款的早期信号</p>
  <p class="meta">基于 Gamalytic 数据 · 2023–2026</p>
</header>

<!-- ════════ NAV ════════ -->
<nav>
  <div class="container">
    <a href="#s0">缘起</a>
    <a href="#s1">前期信号</a>
    <a href="#s2">前照灯</a>
    <a href="#s3">其他发现</a>
    <a href="#s-limit">局限性</a>
  </div>
</nav>

<!-- ════════ SECTION 0: 缘起 ════════ -->
<section id="s0">
<div class="container">
<h2><span class="section-num">0</span>我们想找什么</h2>

<p class="lead">PEAK、R.E.P.O.、幻兽帕鲁——这些项目似乎总是"研发时默默无闻"，"上线后大红大紫"。从业者们对此的反应也各有不同：有人认为爆款肯定每年都有，但是没法提前捕捉；有人认为必须看后台&amp;测试数据才能下准确判断；也有人更相信，还是得靠人对产品的"嗅觉"和"品味"来做判断。</p>

<p>这篇文章的起因是想探索一个问题：真的没有易得数据可以让我们更早捕捉到这些"低调爆款"吗？</p>

<h3>所谓「低调爆款」</h3>
<p>3A大作的成功不在本文的讨论范围内。我更关注那种<strong>超出预期</strong>的惊喜——小团队、没IP、没有大厂背书的"Underdog"，却在上线后炸裂的产品。我从 2023–2026 年间上线的 Steam 游戏中，手工挑选了30 款这样的游戏。</p>

<p>它们有几个共同特征：多数是独立或自发行（63%是自发行），来自全新或小型工作室，上线前没有引起广泛关注，或是不那么被外界看好/甚至也不乏质疑的产品（凑齐30个不容易，且有一定的主观成分在；另一方面，黑神话也被包含进来，部分是因为黑神话在上线前不乏怀疑的声音。且后文我们可以看到，包不包含黑神话，并不影响本文结论）</p>

<div class="chart-container">
<table class="data-table">
<thead><tr><th>游戏</th><th class="num">累计销量<span style="font-weight:normal;color:#94a3b8;font-size:0.75em;">（万份）</span></th><th>发行商</th><th>发售日</th></tr></thead>
<tbody>
"""
    for g in d["success_games"][:30]:
        if g['copies_sold'] >= 1e4:
            copies_str = f"{g['copies_sold']/1e4:.0f}"
        elif g['copies_sold'] > 0:
            copies_str = f"{g['copies_sold']:,}"
        else:
            copies_str = "—"
        html += f'<tr><td><strong>{g["name"]}</strong></td><td class="num">{copies_str}</td><td>{g["publisher"]}</td><td>{g["launch"]}</td></tr>\n'

    html += f"""</tbody></table>
<p class="chart-caption">30款潜力爆款标的，按累计销量排序（销量数据为 Gamalytic 估算值）</p>
</div>

<h3>对照组选取（Backdrop）</h3>
<p>想知道他们如何"异于常人"——我找了一批"常人"来做对比。我从 6万多款 Steam 新品（2023-2026年）中，抽取了 <strong>3,810</strong> 款<strong>有上线前追踪数据</strong>的分层样本，用来模拟一个积极寻找产品的团队在这3年里可能接触到过的产品（6万多款包含大量的业余作品，全都一一看过也不切实际）。这些游戏按关注者数量分层采样：关注者 ≥3,000 的全部纳入，较低关注者段按比例抽样。我们的30款爆款标的也在这个池子里（其中 27 款有足够的上线前数据可供分析）。</p>

<div class="callout">
<strong>数据说明</strong>：Gamalytic 确认，<strong>关注者（Followers）数据来自 Steam 官方 API，是真实数据</strong>；而愿望单（Wishlists）数据是 Gamalytic 的估算值。因此本报告以关注者指标为主要依据，愿望单仅作参考。
</div>

</div>
</section>

<!-- ════════ SECTION 1: 前期信号 ════════ -->
<section id="s1" style="background:var(--card);">
<div class="container">
<h2><span class="section-num">1</span>什么前期信号能区分爆款 vs 一般产品？</h2>

<p class="lead">大家都喜欢重点，这里跳过几千字的分析过程直给结论：重要的区别在于<strong style="color:#dc2626">关注者增长速度</strong>。我们来看一张图：</p>

<div class="chart-container">
<div id="chart_avg_vs_peak" class="chart-box-tall"></div>
<p class="chart-caption">每款游戏的日常增速（中位数）vs 巅峰周增速。绿色 = 爆款（{d["peak_d_comparison"]["n_success"]}款），灰色 = backdrop样本（{d["peak_d_comparison"]["n_backdrop"]}款）。虚线为双指标筛选阈值。</p>
</div>

<ul style="margin:12px 0 20px 24px;">
<li><strong>巅峰周增速（Peak Week Velocity）</strong>：一个游戏在上线前，关注者7日滚动均增最高的一周的数量。通常对应一次重要事件——如预告片发布、试玩节参展、主播带货。它能衡量的是游戏上线前的最高热度怎样？</li>
<li><strong>日常增速（Median Weekly Velocity）</strong>：所有观测周的7日滚动均增的中位数，反映游戏的「日常」吸粉能力。</li>
</ul>

<p>散点图清晰展示了两组的分离：爆款集中在<strong>右上角</strong>（高巅峰增速 + 高日常增速），而大量非爆款分布在左下角或上方偏左（有一两次关注者增长高峰，但日常增速接近于零的「一次性爆发」型选手）。</p>

<p>巅峰周增速衡量最火的一周有多火（爆款是backdrop的{d["peak_d_comparison"]["peak_ratio"]}倍）。日常增速衡量日常热度有多稳（爆款是backdrop的 {d["peak_d_comparison"]["avg_ratio"]}倍）。<strong>爆款组内的产品两者都高，而「一次性爆发」型选手只有高巅峰、低日常</strong></p>

<table class="data-table" style="max-width:650px;">
<thead><tr><th>指标</th><th class="num">爆款中位数</th><th class="num">Backdrop中位数</th><th class="num">倍率</th><th class="num">Cohen's d</th></tr></thead>
<tbody>
<tr><td>巅峰周增速</td><td class="num">{int(d["peak_d_comparison"]["peak_s_median"])}/天</td><td class="num">{int(d["peak_d_comparison"]["peak_b_median"])}/天</td><td class="num">{d["peak_d_comparison"]["peak_ratio"]}x</td><td class="num">{d["peak_d_comparison"]["peak_week_d"]}</td></tr>
<tr><td>日常增速（中位数）</td><td class="num">{int(d["peak_d_comparison"]["avg_s_median"])}/天</td><td class="num">{int(d["peak_d_comparison"]["avg_b_median"])}/天</td><td class="num">{d["peak_d_comparison"]["avg_ratio"]}x</td><td class="num">{d["peak_d_comparison"]["avg_daily_d"]}</td></tr>
</tbody></table>

<div class="callout" style="font-size:0.9em;">
<strong>Cohen's d</strong> 是统计学指标，衡量的是两组数据的「距离」——数值越大，说明爆款和对照组在这个指标上的区分越明显。一般认为，d &gt; 0.8 是「大效应」，d = 0.5 是「中效应」，d = 0.2 是「小效应」。两个指标的 Cohen's d 都超过 1.8（对数尺度）。
</div>

<div class="highlight">
<strong>如何看待这个结果？</strong><br>
<ul style="margin:8px 0 0 20px;">
<li>不意外的部分是大家都知道steam产品的愿望单和关注者数量重要，数量要多，那当然积累得要快？</li>
<li>意外的部分是<strong>增速比绝对数量更能指出爆款潜力。增速捕捉的是Attraction</strong>，商店上展示的pv，一次直播展示的gameplay，一次试玩赋予的体验，吸量和转化能力究竟如何？</li>
</ul>
</div>

<h3>实例看看爆款游戏的增长轨迹长什么样？</h3>

<div class="chart-container">
<div id="chart_trajectories" class="chart-box-tall"></div>
<p class="chart-caption">3款低调爆款的关注者增长轨迹 vs 对照组中位数（灰色）。横轴为距上线天数，纵轴为关注者总数。</p>
</div>

<p>上图中，爆款的关注者数量的水位不一定很高，但轨迹的斜率似乎明显更陡，而且越接近上线越陡。</p>

<h3>实例看看巅峰周增速的区分度</h3>

<div class="chart-container">
<div id="chart_peak_annotated" class="chart-box-tall"></div>
<p class="chart-caption">选定游戏的7日滚动均增轨迹，标注巅峰周位置。灰色虚线为对照组中位数。</p>
</div>

<p>巅峰周可能出现在上线前很久（比如 ARC Raiders 在首次曝光后），也可能出现在上线前几周（比如 Schedule I 在口碑发酵后）。</p>

<h3>实例看看日常增速的区分度</h3>

<div class="chart-container">
<div id="chart_windows" class="chart-box"></div>
<p class="chart-caption">不同时间窗口的关注者周增速中位数对比。爆款从 Steam 页面创建之初就以数倍的速度领先。</p>
</div>

<p>上图指出，日常增速差距从第一天就存在。爆款游戏从 Steam 页面创建之初，日常关注者增速就领先对照组。差距在上线前30天进一步扩大，但即使在上线前一年就已经非常显著。换句话说——<strong>持续热度是一种从早期就可观测的信号。</strong></p>

</div>
</section>

<!-- ════════ SECTION 2: 前照灯 ════════ -->
<section id="s2">
<div class="container">
<h2><span class="section-num">2</span>前照灯——马后炮容易，拿这个指标能筛到好项目吗？</h2>

<p class="lead">在近4000款游戏中，用巅峰周增速和日常增速两个维度，能把爆款筛出来吗？</p>

<h3>门槛定多少？命中率和不漏球率的取舍</h3>

<p>增速门槛定在多少合适呢？这里其实存在取舍：</p>
<ul style="margin:12px 0 20px 24px;">
<li><strong>命中率（Hit Rate）</strong>：「我标记的游戏里，有多少最后成了真正的爆款？」增速门槛定得越高，命中率也越好，但就容易"漏球"。</li>
<li><strong>不漏球率（Catch Rate）</strong>：「在所有最后成功的爆款里，有多少被我提前标记到了？」增速门槛定得越低，越不容易漏球，但也可能标记和关注了一堆"看走眼"的产品。</li>
</ul>

<div class="chart-container">
<div id="chart_threshold" class="chart-box"></div>
<p class="chart-caption">随着巅峰周增速门槛提高（日常增速已做基础过滤），命中率上升但不漏球率下降。</p>
</div>

<h3>三个策略梯度</h3>

<p>我们用<strong>阶梯百分位法</strong>设计了3种策略：巅峰周增速和日常增速各自对应 backdrop 数据集的不同百分位水平。门槛越高，意味着游戏必须在两个维度上都名列前茅。</p>

<table class="data-table">
<thead><tr><th>策略</th><th class="num">巅峰周≥</th><th class="num">日常≥</th><th class="num">关注数</th><th class="num">命中率</th><th class="num">不漏球率</th></tr></thead>
<tbody>
"""
    for t in d["tiers"]:
        html += f'<tr><td><strong>{t["cn_label"]}</strong></td>'
        html += f'<td class="num">{t["peak_threshold"]}/天</td>'
        html += f'<td class="num">{t["avg_threshold"]}/天</td>'
        html += f'<td class="num">{t["radar"]}款</td>'
        html += f'<td class="num">{t["precision"]:.1%}</td>'
        html += f'<td class="num">{t["recall"]:.0%}</td>'
        html += '</tr>\n'

    html += """</tbody></table>

<p>选择一个策略，看看你的关注列表里具体有谁——哪些命中了，哪些漏掉了，哪些是「看走眼？」。</p>

<div class="strategy-tabs" id="strategyTabs">
"""
    for i, t in enumerate(d["tiers"]):
        active = "active" if i == 1 else ""
        html += f'  <button class="{active}" onclick="switchTier({i})">{t["cn_label"]}（巅峰≥{t["peak_threshold"]}+日常≥{t["avg_threshold"]}）</button>\n'

    html += '</div>\n<div class="strategy-panel">\n'

    for i, t in enumerate(d["tiers"]):
        display = "" if i == 1 else "display:none;"
        html += f'<div class="tier-panel" id="tier_{i}" style="{display}">\n'

        html += f"""<div class="strategy-header">
<div class="stat-card"><div class="stat-value" style="color:var(--green)">{t["hit_count"]}/{t["radar"]}</div><div class="stat-label">命中率 {t["precision"]:.1%}</div></div>
<div class="stat-card"><div class="stat-value" style="color:var(--red)">{t["hit_count"]}/{t["evaluable_total"]}</div><div class="stat-label">不漏球率 {t["recall"]:.0%}</div></div>
<div class="stat-card"><div class="stat-value" style="color:var(--amber)">{t["fp_count"]}</div><div class="stat-label">看走眼？</div></div>
<div class="stat-card"><div class="stat-value">{t["radar"]}</div><div class="stat-label">关注列表</div></div>
</div>
"""
        html += '<div class="strategy-columns">\n'

        html += '<div class="strategy-col"><div class="strategy-col-header" style="background:var(--green);">'
        html += f'命中（{t["hit_count"]}）</div><div class="strategy-col-body">\n'
        for g in t["hits"]:
            html += f'<div class="game-item"><span>{g["name"]}</span><span class="game-vel">{int(g["peak"])}/天</span></div>\n'
        html += '</div></div>\n'

        html += '<div class="strategy-col"><div class="strategy-col-header" style="background:var(--red);">'
        html += f'遗漏（{t["miss_count"]} + {t["no_data_count"]}无数据）</div><div class="strategy-col-body">\n'
        for g in t["misses"]:
            html += f'<div class="game-item"><span>{g["name"]}</span><span class="game-vel">{int(g["peak"])}/天</span></div>\n'
        if t["no_data"]:
            html += '<div class="divider">── 无上线前数据 ──</div>\n'
            for name in t["no_data"]:
                html += f'<div class="game-item"><span>{name}</span><span class="game-vel">—</span></div>\n'
        html += '</div></div>\n'

        html += '<div class="strategy-col"><div class="strategy-col-header" style="background:var(--amber);">'
        html += f'看走眼？（{t["fp_count"]}）</div><div class="strategy-col-body">\n'
        for g in t["fps"][:50]:
            html += f'<div class="game-item"><span>{g["name"]}</span><span class="game-vel">{int(g["peak"])}/天</span></div>\n'
        if len(t["fps"]) > 50:
            html += f'<div class="game-item" style="color:var(--gray)">… 及其他 {len(t["fps"])-50} 款</div>\n'
        html += '</div></div>\n'

        html += '<div class="strategy-col"><div class="strategy-col-header" style="background:#475569;">'
        html += f'完整关注列表（{t["radar"]}）</div><div class="strategy-col-body">\n'
        for g in t["watchlist"][:80]:
            color = "var(--green)" if g["group"] == "success" else "var(--gray)"
            html += f'<div class="game-item"><span style="color:{color}">{g["name"]}</span><span class="game-vel">{int(g["peak"])}/天</span></div>\n'
        if len(t["watchlist"]) > 80:
            html += f'<div class="game-item" style="color:var(--gray)">… 及其他 {len(t["watchlist"])-80} 款</div>\n'
        html += '</div></div>\n'

        html += '</div>\n'
        html += '</div>\n'

    html += '</div>\n'

    # ── Strategy Quality Section ──
    sq = d["strategy_quality"]
    html += f"""

<h3>19%命中率+41%不漏球率，这个策略能让人满意么？</h3>

<p>说19%的命中率很高似乎有点勉强（似乎有很多无用功和看走眼！）。但核心原因在于我们这个指标只聚焦于命中 30 个「意外爆款」。视野如果打开一点呢？</p>

<p>换一个务实的视角——<strong>用这些策略挑选的游戏，整体商业表现如何？</strong></p>

<div class="chart-container">
<div id="chart_strategy_quality" class="chart-box"></div>
<p class="chart-caption">三种策略下的组合质量。收入 &gt;$10M 和 &gt;$50M 占比、进入全样本收入 Top100 的占比。</p>
</div>

<p>如果把目标换成"<strong>找到收入 &gt;$50M 的高商业价值游戏</strong>"（样本池中共 {sq["total_50m"]} 款），三种策略的命中率和不漏球率如何？</p>

<div style="overflow-x:auto;">
<table class="data-table">
<thead><tr><th>策略</th><th class="num">命中/关注数</th><th class="num">命中率</th><th class="num">不漏球率</th><th class="num">看走眼</th><th class="num">收入中位数</th><th class="num">进入Top100</th></tr></thead>
<tbody>
"""
    for t in sq["tiers"]:
        html += f'<tr><td><strong>{t["label"]}</strong></td>'
        html += f'<td class="num">{t["n_50m"]}/{t["n"]}</td>'
        html += f'<td class="num">{t["hit_50m_precision"]:.0%}</td>'
        html += f'<td class="num">{t["hit_50m_recall"]:.0%}</td>'
        html += f'<td class="num">{t["fps_50m"]}款</td>'
        html += f'<td class="num">${t["median_rev"]}M</td>'
        html += f'<td class="num">{t["n_top100"]}/{t["n"]}</td></tr>\n'

    html += f"""</tbody></table>
</div>
<p style="color:#94a3b8;font-size:0.8rem;margin-top:4px;">* 进入Top100 指 {sq["total_games"]:,} 款样本池中，按（累计）营收排名前 100 的游戏（门槛约 ${sq["top100_threshold"]}M）。</p>

<p>高确信策略中收入中位数达 <strong>${sq["tiers"][2]["median_rev"]}M</strong>，超过一半是 $50M+ 的高商业价值游戏；即使是最宽松的广角策略，收入中位数也有 ${sq["tiers"][0]["median_rev"]}M，且 <strong>93% 的 $50M+ 游戏都不会漏掉</strong>。这个指标对高商业价值游戏的命中率和不漏球率看起来相当不错。</p>

"""

    det50 = d["detection_50m"]
    # Build short-data game names for the footnote
    short_names = "、".join(cn(g["name"]) for g in det50["short_data_games"]) if det50["short_data_games"] else ""
    html += f"""

<h3>标准越高，发现越晚</h3>
<p>门槛更高意味着更精准，但也意味着进入雷达的时间越晚。三种策略识别高商业价值产品（收入&gt;$50M）的提前量是怎样的分布？</p>

<div class="chart-container">
<div id="chart_detection_box" class="chart-box" style="height:420px;"></div>
<p class="chart-caption">对 {det50["total_50m"]} 款收入 &gt;$50M 的游戏，三种策略首次检测到的提前天数分布（箱线图）。虚线标注 30 天、90 天、270 天参考线。</p>
</div>

<p>考虑到「日常增速中位数」需要足够的样本量才有意义，这里我们至少要求有 <strong>{det50["min_n"]} 个样本</strong>以上才可达标。在这个要求下，个别上线前不满 {det50["min_n"]} 天数据的产品（{short_names}）无法被纳入监测——但凭借其极高的巅峰增速和已知的品牌背书，似乎也不难做判断。</p>

<p>广角策略中位提前量 <strong>{int(det50["tiers"][0]["median"])} 天</strong>，能捕获 {det50["tiers"][0]["detected"]}/{det50["total_50m"]} 款；高确信策略中位提前量 <strong>{int(det50["tiers"][2]["median"])} 天</strong>，但仅能触发 {det50["tiers"][2]["detected"]}/{det50["total_50m"]} 款。<strong>低门槛的早期监控 + 高门槛的后期确认</strong>可能是比单一策略更好的组合方式。</p>

<h3>投入-回报角度看：每发现一个 $10M+ 的产品，需要去看多少款？</h3>

<table class="data-table" style="max-width:500px;">
<thead><tr><th>策略</th><th class="num">关注数</th><th class="num">$10M+产品</th><th class="num">每发现1个需看</th></tr></thead>
<tbody>
"""
    for t in sq["tiers"]:
        html += f'<tr><td><strong>{t["label"]}</strong></td>'
        html += f'<td class="num">{t["n"]}款</td>'
        html += f'<td class="num">{t["n_10m"]}款</td>'
        html += f'<td class="num">{t["games_per_10m"]}款</td></tr>\n'

    html += f"""</tbody></table>

<p>精选策略每看 <strong>{sq["tiers"][1]["games_per_10m"]}</strong> 款游戏就能发现一个收入超 $10M 的产品；高确信策略更是每 <strong>{sq["tiers"][2]["games_per_10m"]}</strong> 款就有一个——按照这个标准看，似乎不存在「浪费精力」的问题。</p>

<div class="highlight">
<strong>关键结论：这些策略可能无法一五一十精确命中文章一开始举出的30个案例，但它们在商业层面似乎是有效的。</strong>用"精选策略"举例而言：长期寻找那些巅峰周增长了7000+关注，日常比较一致地自然增长300+关注的产品，则可以平均提前半年，以35%命中率，识别到77%的高商业价值产品。收入中位数在$36M。对于需要看产品的游戏行业从业者，<strong>关注者增速</strong>都是一个成本低、信号强的前向雷达。
</div>
</div>
</section>
"""

    html += """
<!-- ════════ SECTION 3: 其他发现 ════════ -->
<section id="s3" style="background:var(--card);">
<div class="container">
<h2><span class="section-num">3</span>其他发现</h2>

<h3>这个策略闻不到的爆款</h3>

<table class="data-table">
<thead><tr><th>游戏</th><th class="num">巅峰周增速</th><th class="num">日常增速中位数</th><th class="num">首月营收</th></tr></thead>
<tbody>
<tr><td>致命公司</td><td class="num">57/天</td><td class="num">2/天</td><td class="num">$146M</td></tr>
<tr><td>恶魔轮盘</td><td class="num">269/天</td><td class="num">152/天</td><td class="num">$17M</td></tr>
<tr><td>小丑牌</td><td class="num">214/天</td><td class="num">26/天</td><td class="num">$78M</td></tr>
<tr><td>Megabonk</td><td class="num">87/天</td><td class="num">8/天</td><td class="num">$32M</td></tr>
</tbody>
</table>

<h3>爆款vs一般产品「没什么差别」的指标</h3>
<p>几个没有区分度的指标：</p>

<table class="data-table">
<thead><tr><th>指标</th><th>Cohen's d</th><th>意味着什么</th></tr></thead>
<tbody>
<tr><td>售价</td><td>0.09</td><td>什么价格都能爆</td></tr>
<tr><td>开发者历史营收</td><td>0.01</td><td>新工作室和老工作室一样可能出爆款</td></tr>
<tr><td>上线前挂页天数</td><td>0.08</td><td>挂半年和挂两年都行</td></tr>
<tr><td>标签数量</td><td>0.25</td><td>给自己打满标签也用处不大</td></tr>
</tbody>
</table>

<p>值得一提的是<strong>开发者历史（d=0.01）</strong>它几乎等于零，可能也是我们选择的这组产品导致的结果：它们就是来自游戏行业之前的小透明，也进一步增加了这些产品上线后表现令人"意外"的程度。致命公司的开发者 Zeekerss 之前只有一些小众的作品；小丑牌的 LocalThunk 也是第一次做游戏。似乎很难从开发者的简历预测爆款。</p>

</div>
</section>

<!-- ════════ SECTION: 局限性 ════════ -->
<section id="s-limit">
<div class="container">
<h2>研究局限性</h2>

<p>这份研究中，仍有较多的局限性：</p>

<ol style="margin:12px 0 20px 24px;">
<li><strong>Gamalytic 追踪缺口</strong>：有3款爆款（PEAK、Dark and Darker、Content Warning）因 Steam 页面创建极晚或 Gamalytic 开始追踪较晚，没有足够的上线前关注者数据（&lt;7天）。</li>
<li><strong>前向评估样本的局限</strong>：策略评估基于 Gamalytic 有上线前追踪的 ~3,800 款游戏样本。这些游戏偏向有一定关注度的产品（followers ≥ 500），不完全代表 Steam 全量 6 万+游戏。</li>
<li><strong>无因果推断</strong>：高关注者增速可能是病毒传播的<em>结果</em>（TikTok 带火 → 关注者暴涨），而非<em>原因</em>。不存在"因为这个产品Follower增长很快所以必然大卖"这样的因果关系</li>
<li><strong>营收/销量仍为估算值</strong>：只有关注者数据经 Gamalytic 确认为 Steam 真实数据，营收和销量数据均为算法估算。</li>
</ol>

</div>
</section>

<footer>
<p>慧眼识珠：Steam爆款循迹？ · 数据来源：Gamalytic API · 2026年3月</p>
<p>30款爆款 · """ + f'{d["backdrop_stats"]["sample_size"]:,}' + """款backdrop样本 · 关注者数据为 Steam 真实数据</p>
</footer>

<script>
// ── Chart Data ──
const trajectoryData = """ + json.dumps(d["trajectories"], ensure_ascii=False) + """;
const controlEnvelope = """ + json.dumps(d["control_envelope"], ensure_ascii=False) + """;
const peakDistributionFull = """ + json.dumps(d["peak_distribution_full"], ensure_ascii=False) + """;
const peakAnnotated = """ + json.dumps(d["peak_annotated"], ensure_ascii=False) + """;
const thresholdData = """ + json.dumps(d["threshold_sweep"], ensure_ascii=False) + """;
const windowData = """ + json.dumps(d["velocity_windows"], ensure_ascii=False) + """;
const stratQuality = """ + json.dumps(d["strategy_quality"], ensure_ascii=False) + """;
const detection50m = """ + json.dumps(d["detection_50m"]["tiers"], ensure_ascii=False) + """;

// ── Chart Theme ──
const colors = {green:'#10b981', gray:'#94a3b8', blue:'#3b82f6', amber:'#f59e0b', red:'#ef4444', primary:'#1b2838'};

// ── Utility ──
var _charts = {};
function initChart(id) {
  var el = document.getElementById(id);
  if (!el) return null;
  var c = echarts.init(el);
  _charts[id] = c;
  return c;
}

// ── Chart 2: Trajectories ──
(function() {
  var c = initChart('chart_trajectories');
  if (!c) return;
  var series = [];
  // Control envelope
  series.push({
    name:'对照组中位数', type:'line', data: controlEnvelope, smooth:true,
    lineStyle:{color:colors.gray, width:2, type:'dashed'}, symbol:'none',
    itemStyle:{color:colors.gray},
    areaStyle:{color:'rgba(148,163,184,0.08)'},
  });
  // Selected games
  var gameColors = ['#10b981','#3b82f6','#f59e0b','#ef4444','#8b5cf6'];
  var idx = 0;
  for (var name in trajectoryData) {
    series.push({
      name: name, type:'line', data: trajectoryData[name], smooth:true,
      lineStyle:{color:gameColors[idx%5], width:2.5}, symbol:'none',
      itemStyle:{color:gameColors[idx%5]},
    });
    idx++;
  }
  c.setOption({
    tooltip: {trigger:'axis'},
    legend: {bottom:0, textStyle:{fontSize:12}},
    grid: {left:80, right:30, top:30, bottom:75},
    xAxis: {type:'value', name:'距上线天数', nameLocation:'middle', nameGap:30,
      inverse:true, min:-7, max:200,
      axisLabel:{formatter:function(v){return v>0?'T-'+v:v===0?'上线':'T+'+Math.abs(v);}}},
    yAxis: {type:'value', nameLocation:'end', nameGap:15, nameTextStyle:{padding:[0,0,0,50]},
      axisLabel:{formatter:function(v){return v>=1000?Math.round(v/1000)+'K':v;}}},
    series: series,
  });
})();

// ── Chart 2b: Avg vs Peak Scatter (full backdrop) ──
(function() {
  var c = initChart('chart_avg_vs_peak');
  if (!c) return;
  var sData = peakDistributionFull.filter(g=>g.group==='success');
  var bData = peakDistributionFull.filter(g=>g.group!=='success');
  // Confidence ellipse in log-space
  function confEllipse(data, xKey, yKey, nPts, sigma) {
    var pts = data.filter(d=>d[xKey]>0&&d[yKey]>0);
    if (pts.length < 3) return [];
    var n = pts.length, mx=0, my=0;
    pts.forEach(function(d){mx+=Math.log10(Math.max(d[xKey],0.3));my+=Math.log10(d[yKey]);});
    mx/=n; my/=n;
    var cxx=0, cyy=0, cxy=0;
    pts.forEach(function(d){
      var dx=Math.log10(Math.max(d[xKey],0.3))-mx, dy=Math.log10(d[yKey])-my;
      cxx+=dx*dx; cyy+=dy*dy; cxy+=dx*dy;
    });
    cxx/=(n-1); cyy/=(n-1); cxy/=(n-1);
    var tr=cxx+cyy, det=cxx*cyy-cxy*cxy;
    var disc=Math.sqrt(Math.max(tr*tr/4-det,0));
    var l1=tr/2+disc, l2=Math.max(tr/2-disc,0.001);
    var ang=Math.atan2(2*cxy, cxx-cyy)/2;
    var a=sigma*Math.sqrt(l1), b=sigma*Math.sqrt(l2);
    var result=[];
    for(var i=0;i<=nPts;i++){
      var t=2*Math.PI*i/nPts;
      var ex=a*Math.cos(t), ey=b*Math.sin(t);
      var rx=ex*Math.cos(ang)-ey*Math.sin(ang)+mx;
      var ry=ex*Math.sin(ang)+ey*Math.cos(ang)+my;
      result.push([Math.pow(10,rx),Math.pow(10,ry)]);
    }
    return result;
  }
  var sEllipse = confEllipse(sData,'avg_daily','peak_week',80,2);
  var bEllipse = confEllipse(bData,'avg_daily','peak_week',80,2);
  c.setOption({
    tooltip: {trigger:'item', formatter: function(p) {
      var d = p.data;
      if (!d[2]) return '';
      return '<b>'+d[2]+'</b><br>日常增速（中位数）: '+Math.round(d[0])+'/天<br>巅峰周增速: '+Math.round(d[1])+'/天';
    }},
    legend: {data:['爆款','Backdrop样本'], bottom:0},
    grid: {left:70, right:40, top:40, bottom:70, containLabel:true},
    xAxis: {type:'log', name:'日常增速（中位数，人/天）', nameLocation:'middle', nameGap:28,
      min:0.3, axisLabel:{formatter:function(v){return v>=1000?Math.round(v/1000)+'K':v;}}},
    yAxis: {type:'log', name:'巅峰周增速（人/天）', nameLocation:'end', nameGap:15,
      min:1, axisLabel:{formatter:function(v){return v>=1000?Math.round(v/1000)+'K':v;}}},
    series: [
      // Backdrop ellipse (gray fill)
      {name:'Backdrop样本', type:'line', data:bEllipse, smooth:true, symbol:'none',
        lineStyle:{color:colors.gray, width:1.5, type:'dashed'},
        areaStyle:{color:'rgba(148,163,184,0.10)'}, silent:true, z:1},
      // Success ellipse (green fill)
      {name:'爆款', type:'line', data:sEllipse, smooth:true, symbol:'none',
        lineStyle:{color:colors.green, width:1.5, type:'dashed'},
        areaStyle:{color:'rgba(16,185,129,0.12)'}, silent:true, z:1},
      // Backdrop scatter
      {name:'Backdrop样本', type:'scatter',
        data:bData.filter(d=>d.avg_daily>0).map(d=>[Math.max(d.avg_daily,0.3),d.peak_week,d.name]),
        symbolSize:4, itemStyle:{color:colors.gray, opacity:0.25}, z:2},
      // Success scatter with labels
      {name:'爆款', type:'scatter',
        data:sData.filter(d=>d.avg_daily>0).map(d=>[Math.max(d.avg_daily,0.3),d.peak_week,d.name]),
        symbolSize:12, itemStyle:{color:colors.green, opacity:0.9}, z:3,
        label:{show:true, formatter:function(p){return p.data[2];}, fontSize:8, position:'right', color:'#065f46'},
        markLine: {silent:true, symbol:'none', lineStyle:{color:'#b0b8c4', type:'dashed', width:1},
          data:[
            {yAxis:900, label:{formatter:'巅峰周≥900', position:'insideEndTop', fontSize:9, color:'#94a3b8'}},
            {xAxis:40, label:{formatter:'日常≥40', position:'insideEndTop', fontSize:9, color:'#94a3b8'}},
          ]}
      },
    ],
  });
})();

// ── Chart: Strategy Quality ──
(function() {
  var c = initChart('chart_strategy_quality');
  if (!c) return;
  var tiers = stratQuality.tiers;
  var labels = tiers.map(t=>t.label);
  c.setOption({
    tooltip: {trigger:'axis'},
    legend: {data:['收入>$10M占比','收入>$50M占比','进入Top100占比'], bottom:0},
    grid: {left:60, right:30, top:30, bottom:60},
    xAxis: {type:'category', data:labels},
    yAxis: {type:'value', max:1, axisLabel:{formatter:function(v){return (v*100)+'%';}}},
    series: [
      {name:'收入>$10M占比', type:'bar', data:tiers.map(t=>t.pct_10m),
        itemStyle:{color:colors.green}, barGap:'15%',
        label:{show:true, position:'top', fontSize:11, formatter:function(p){return Math.round(p.value*100)+'%';}}},
      {name:'收入>$50M占比', type:'bar', data:tiers.map(t=>t.pct_50m),
        itemStyle:{color:colors.blue},
        label:{show:true, position:'top', fontSize:11, formatter:function(p){return Math.round(p.value*100)+'%';}}},
      {name:'进入Top100占比', type:'bar', data:tiers.map(t=>t.pct_top100),
        itemStyle:{color:colors.amber},
        label:{show:true, position:'top', fontSize:11, formatter:function(p){return Math.round(p.value*100)+'%';}}},
    ]
  });
})();

// ── Chart: Detection Lead Time Boxplot ($50M+) ──
(function() {
  var c = initChart('chart_detection_box');
  if (!c) return;

  var labels = detection50m.map(function(t) { return t.label; });
  var tierColors = detection50m.map(function(t) { return t.color; });

  // Boxplot data: [min, Q1, median, Q3, max]
  var boxData = detection50m.map(function(d, i) {
    return {value:[d.min, d.q1, d.median, d.q3, d.max],
      itemStyle:{borderColor:tierColors[i], color:tierColors[i]+'30'}};
  });

  // Scatter: individual points with jitter, including game names for tooltip
  var scatterSeries = detection50m.map(function(t, i) {
    return {
      name:t.label, type:'scatter',
      data:t.values.map(function(v, j){ return {value:[i+(Math.random()-0.5)*0.25, v], gameName:t.names[j]}; }),
      itemStyle:{color:t.color, opacity:0.6}, symbolSize:6, z:3,
    };
  });

  // Use convertToPixel after first render to align graphic cards with x-axis
  // For now use grid-relative percentage; we'll adjust after setOption
  var gridLeft = 65, gridRight = 30;

  c.setOption({
    tooltip:{trigger:'item', formatter:function(p){
      if(p.seriesType==='boxplot'){
        var v=p.value;
        return p.name+'<br>最大: '+Math.round(v[4])+'天<br>Q3: '+Math.round(v[3])+'天<br>中位: '+Math.round(v[2])+'天<br>Q1: '+Math.round(v[1])+'天<br>最小: '+Math.round(v[0])+'天';
      }
      var gn=p.data.gameName||'';
      return gn+'<br>提前 '+Math.round(p.value[1])+' 天发现';
    }},
    grid:{left:gridLeft, right:gridRight, top:65, bottom:40},
    xAxis:{type:'category', data:labels, axisLabel:{fontSize:13, fontWeight:'bold',
      color:function(v,i){return tierColors[i]||'#94a3b8';}}},
    yAxis:{type:'value', name:'提前发现天数', nameLocation:'middle', nameGap:45,
      min:0, axisLabel:{formatter:function(v){return v+'天';}},
      splitLine:{lineStyle:{color:'rgba(148,163,184,0.15)'}}},
    series:[{
      name:'分布', type:'boxplot', data:boxData, boxWidth:['30%','50%'],
      markLine:{
        silent:true, symbol:'none',
        lineStyle:{type:'dashed', width:1},
        label:{position:'insideEndTop', fontSize:12},
        data:[
          {yAxis:30, lineStyle:{color:'#f59e0b'}, label:{formatter:'30天', color:'#f59e0b'}},
          {yAxis:90, lineStyle:{color:'#f59e0b'}, label:{formatter:'90天', color:'#f59e0b'}},
          {yAxis:270, lineStyle:{color:'#f59e0b'}, label:{formatter:'270天', color:'#f59e0b'}},
        ]
      },
    }].concat(scatterSeries),
  });

  // Position summary cards aligned with each category using convertToPixel
  function buildCards(){
    var graphics = [];
    detection50m.forEach(function(t, i){
      var px = c.convertToPixel({xAxisIndex:0}, i);
      if(!px) return;
      graphics.push(
        {type:'text', position:[px, 8], style:{text:t.detected+'/'+t.total+'款命中', fill:t.color,
          font:'bold 13px sans-serif', textAlign:'center'}},
        {type:'text', position:[px, 28], style:{text:'中位 '+Math.round(t.median)+'天 | 均值 '+Math.round(t.mean)+'天',
          fill:'#333', font:'bold 13px sans-serif', textAlign:'center'}}
      );
    });
    c.setOption({graphic:graphics});
  }
  setTimeout(buildCards, 100);
  window.addEventListener('resize', function(){ setTimeout(buildCards, 150); });
})();

// ── Chart 4: Peak Annotated Trajectories ──
(function() {
  var c = initChart('chart_peak_annotated');
  if (!c) return;
  var series = [];
  var annotations = [];
  var colorMap = {'漫威争锋':'#10b981','ARC Raiders':'#3b82f6','Manor Lords':'#f59e0b','Schedule I':'#ef4444'};
  var fallbackColors = ['#10b981','#3b82f6','#f59e0b','#ef4444'];
  peakAnnotated.forEach(function(game, idx) {
    var clr = colorMap[game.name] || fallbackColors[idx%4];
    series.push({
      name: game.name, type:'line', data: game.data, smooth:true,
      lineStyle:{color:clr, width:2}, symbol:'none',
      itemStyle:{color:clr},
    });
    // Mark peak
    if (game.peak_dtl) {
      annotations.push({
        name: game.name+' 巅峰', coord: [game.peak_dtl, game.peak_week],
        symbol: 'pin', symbolSize: 30,
        itemStyle: {color: clr},
        label: {show:true, formatter:Math.round(game.peak_week)+'/天', fontSize:10, position:'top'},
      });
    }
  });
  c.setOption({
    tooltip: {trigger:'axis', formatter: function(params) {
      var s = 'T-'+params[0].value[0]+' 天<br>';
      params.forEach(function(p){if(p.value[1]!=null)s+=p.marker+p.seriesName+': '+Math.round(p.value[1])+'/天<br>';});
      return s;
    }},
    legend: {bottom:5},
    grid: {left:80, right:30, top:30, bottom:70},
    xAxis: {type:'value', name:'距上线天数', inverse:true, min:0, max:300, nameLocation:'middle', nameGap:35,
      axisLabel:{formatter:function(v){return 'T-'+v;}}},
    yAxis: {type:'value', nameLocation:'end', nameGap:15, nameTextStyle:{padding:[0,0,0,80]},
      axisLabel:{formatter:function(v){return v>=1000?Math.round(v/1000)+'K':v;}}},
    series: series.concat([{type:'scatter',data:annotations.map(a=>a.coord),
      symbolSize:0, itemStyle:{opacity:0}, markPoint:{data:annotations},
      silent:true, z:10}]),
  });
})();

// ── Chart 5: Threshold Sweep ──
(function() {
  var c = initChart('chart_threshold');
  if (!c) return;
  c.setOption({
    tooltip: {trigger:'item', formatter: function(p) {
      var d = p.data;
      if (!d || d.length < 2) return '';
      return p.marker+p.seriesName+': '+(d[1]*100).toFixed(1)+'%<br>门槛: ≥'+Math.round(d[0])+'/天';
    }},
    legend: {data:['命中率','不漏球率'], bottom:0},
    grid: {left:60, right:30, top:40, bottom:70},
    xAxis: {type:'log', name:'巅峰周增速门槛（人/天）', nameLocation:'middle', nameGap:28,
      min:50, max:10000},
    yAxis: {type:'value', name:'比率', nameLocation:'end', nameGap:15, max:1,
      axisLabel:{formatter:function(v){return (v*100)+'%';}}},
    series: [
      {name:'命中率', type:'line', data:thresholdData.map(d=>[d.threshold,d.precision]),
        lineStyle:{color:colors.green, width:2.5}, symbol:'none', smooth:true, itemStyle:{color:colors.green},
        markLine: {silent:true, symbol:'none', data:[
          {xAxis:400, lineStyle:{color:colors.green, type:'dashed', width:1.5},
            label:{formatter:'广角 400\\n命中6%', fontSize:9, color:colors.green, lineHeight:14}},
          {xAxis:900, lineStyle:{color:colors.amber, type:'dashed', width:1.5},
            label:{formatter:'精选 900\\n命中13%', fontSize:9, color:colors.amber, lineHeight:14}},
          {xAxis:1900, lineStyle:{color:colors.red, type:'dashed', width:1.5},
            label:{formatter:'高确信 1900\\n命中19%', fontSize:9, color:colors.red, lineHeight:14}},
        ]}},
      {name:'不漏球率', type:'line', data:thresholdData.map(d=>[d.threshold,d.recall]),
        lineStyle:{color:colors.blue, width:2.5}, symbol:'none', smooth:true, itemStyle:{color:colors.blue}},
    ],
  });
})();

// ── Chart 7: Velocity Windows ──
(function() {
  var c = initChart('chart_windows');
  if (!c) return;
  var windows = ['T-360→T-180','T-180→T-90','T-90→T-30','T-30→上线'];
  var sVals = windows.map(w => {var d=windowData.find(d=>d.window===w&&d.group==='success'); return d?d.median:0;});
  var cVals = windows.map(w => {var d=windowData.find(d=>d.window===w&&d.group==='control'); return d?d.median:0;});
  var ratios = sVals.map((s,i) => cVals[i]>0 ? (s/cVals[i]).toFixed(1)+'x' : '-');
  c.setOption({
    tooltip: {trigger:'axis'},
    legend: {data:['爆款','对照组'], bottom:0},
    grid: {left:70, right:30, top:30, bottom:50},
    xAxis: {type:'category', data:windows},
    yAxis: {type:'value', name:'中位增速（人/天）'},
    series: [
      {name:'爆款', type:'bar', data:sVals, itemStyle:{color:colors.green}, barGap:'20%',
        label:{show:true, position:'top', fontSize:10, formatter:function(p){return Math.round(p.value);}}},
      {name:'对照组', type:'bar', data:cVals, itemStyle:{color:colors.gray},
        label:{show:true, position:'top', fontSize:10, formatter:function(p){return Math.round(p.value);}}},
    ],
    graphic: ratios.map(function(r, i) {
      return {type:'text', left: (15 + i*22)+'%', top:8, style:{text:r+'倍', fill:'#ef4444', fontSize:12, fontWeight:'bold'}};
    })
  });
})();

// ── Strategy Tab Switching ──
function switchTier(idx) {
  document.querySelectorAll('.tier-panel').forEach(function(p,i){p.style.display=i===idx?'':'none';});
  document.querySelectorAll('.strategy-tabs button').forEach(function(b,i){b.className=i===idx?'active':'';});
}

// ── Nav active state ──
var sections = document.querySelectorAll('section[id]');
window.addEventListener('scroll', function() {
  var scrollPos = window.scrollY + 100;
  sections.forEach(function(s) {
    if (s.offsetTop <= scrollPos && s.offsetTop + s.offsetHeight > scrollPos) {
      document.querySelectorAll('nav a').forEach(function(a){a.className='';});
      var link = document.querySelector('nav a[href="#'+s.id+'"]');
      if (link) link.className = 'active';
    }
  });
});

// ── Mobile Responsive ──
(function() {
  var MOB = 600;
  var wasMob = window.innerWidth <= MOB;

  var mobOpt = {
    chart_trajectories: {
      tooltip:{confine:true},
      grid:{left:50,right:15,top:20,bottom:55},
      legend:{textStyle:{fontSize:10}}
    },
    chart_avg_vs_peak: {
      tooltip:{confine:true},
      grid:{left:35,right:15,top:25,bottom:50,containLabel:true},
      xAxis:{nameGap:20,axisLabel:{fontSize:10}},
      yAxis:{nameGap:8,axisLabel:{fontSize:10}},
      series:[{},{},{},{label:{show:false}}]
    },
    chart_strategy_quality: {
      tooltip:{confine:true},
      grid:{left:40,right:15,top:20,bottom:45},
      legend:{textStyle:{fontSize:10}},
      series:[{label:{fontSize:9}},{label:{fontSize:9}},{label:{fontSize:9}}]
    },
    chart_peak_annotated: {
      tooltip:{confine:true},
      grid:{left:50,right:15,top:20,bottom:65},
      legend:{textStyle:{fontSize:10},bottom:0}
    },
    chart_threshold: {
      tooltip:{confine:true},
      grid:{left:45,right:15,top:45,bottom:75},
      series:[{markLine:{label:{fontSize:7}}}]
    },
    chart_windows: {
      tooltip:{confine:true},
      grid:{left:45,right:15,top:40,bottom:40},
      yAxis:{name:''},
      xAxis:{axisLabel:{fontSize:10}},
      series:[{label:{fontSize:8}},{label:{fontSize:8}}]
    },
  };

  var deskOpt = {
    chart_trajectories: {
      tooltip:{confine:false},
      grid:{left:80,right:30,top:30,bottom:75},
      legend:{textStyle:{fontSize:12}}
    },
    chart_avg_vs_peak: {
      tooltip:{confine:false},
      grid:{left:70,right:40,top:40,bottom:70},
      xAxis:{nameGap:28,axisLabel:{fontSize:12}},
      yAxis:{nameGap:15,axisLabel:{fontSize:12}},
      series:[{},{},{},{label:{show:true}}]
    },
    chart_strategy_quality: {
      tooltip:{confine:false},
      grid:{left:60,right:30,top:30,bottom:60},
      legend:{textStyle:{fontSize:12}},
      series:[{label:{fontSize:11}},{label:{fontSize:11}},{label:{fontSize:11}}]
    },
    chart_peak_annotated: {
      tooltip:{confine:false},
      grid:{left:80,right:30,top:30,bottom:70},
      legend:{textStyle:{fontSize:12}}
    },
    chart_threshold: {
      tooltip:{confine:false},
      grid:{left:60,right:30,top:40,bottom:70}
    },
    chart_windows: {
      tooltip:{confine:false},
      grid:{left:70,right:30,top:30,bottom:50},
      yAxis:{name:'中位增速（人/天）'},
      xAxis:{axisLabel:{fontSize:12}},
      series:[{label:{fontSize:10}},{label:{fontSize:10}}]
    },
  };

  function applyMode(mobile) {
    var opts = mobile ? mobOpt : deskOpt;
    Object.keys(opts).forEach(function(id) {
      if (_charts[id]) _charts[id].setOption(opts[id]);
    });
  }

  if (wasMob) applyMode(true);

  window.addEventListener('resize', function() {
    Object.keys(_charts).forEach(function(id) {
      if (_charts[id]) _charts[id].resize();
    });
    var nowMob = window.innerWidth <= MOB;
    if (nowMob !== wasMob) {
      wasMob = nowMob;
      applyMode(nowMob);
    }
  });
})();
</script>
</body>
</html>"""
    return html


# ── MD Generation ──────────────────────────────────────────────────────

def generate_md(report_data):
    d = report_data
    md = """# 「慧眼识珠」：Steam爆款是否有迹可循？

## 从真实数据出发，寻找下一个爆款的早期信号

从 Steam 真实数据出发，寻找下一个爆款的早期信号

基于 Gamalytic 数据 · 30款爆款 vs 300款对照 · 2023–2026

---

## 第零章：我们想找什么

PEAK、R.E.P.O.、幻兽帕鲁——这些上线前看似「默默无闻」的项目，为什么能在发售后一飞冲天？如果时光倒流，我们能不能从公开数据中提前嗅到爆款的味道？换句话说：这些游戏是不是年纪轻轻就「骨骼清奇」？

带着这个好奇心，我们做了一次系统性的研究。

### 什么是「潜力爆款」？

我们关注的不是3A大作的必然成功，而是那种**超出预期**的惊喜——小团队、新IP、没有大厂背书，却在上线后炸裂的产品。我们从 2023–2026 年间的 Steam 游戏中，手工挑选了 **30 款**这样的标的。

它们有几个共同特征：多数是独立或自发行（63%是自发行），来自全新或小型工作室，上线前没人预料到它们会成为现象级产品。

[图表：30款爆款标的列表]

### 对照组：它们的「同龄人」

光有爆款还不够——要发现「骨骼清奇」的特征，还得有一群「正常孩子」做对比。我们用标签相似度（Jaccard, 权重60%）、上线日期（20%）和价格（20%）三重匹配，为每款爆款找了约10个同类产品，共 **300 款对照组游戏**。

对照组都是5万份以上销量的产品——不是失败作品，而是「表现还行」的游戏。所以我们比较的是「爆炸级成功」vs「普通成功」，这让任何发现都更有说服力：能在都还不错的游戏里脱颖而出的指标，才是真正有价值的信号。

> **数据可靠性说明**：Gamalytic 确认，**关注者（Followers）数据来自 Steam 官方 API，是真实数据**；而愿望单（Wishlists）数据是 Gamalytic 的估算值。因此本报告以关注者指标为主要依据，愿望单仅作参考。

---

## 第一章：后视镜——数据里的初步发现

Gamalytic 提供了丰富的历史数据：关注者数、愿望单数、评测数、销量、营收……但考虑到我们的目标是「上线前就识别」，可用的指标迅速收敛到了上线前能观测的几个：关注者增速、愿望单增速、上线时的绝对数值。

那么问题来了：在这些指标里，**哪个最能区分爆款和普通游戏？**

我们使用 **Cohen's d** 来回答这个问题。简单说，Cohen's d 衡量的是两组数据的「距离」——数值越大，说明爆款和对照组在这个指标上的差异越明显。一般认为，d > 0.8 是「大效应」，d = 0.5 是「中效应」，d = 0.2 是「小效应」。

[图表：各指标的 Cohen's d 排名]

"""
    md += f"""> **核心发现：关注者增速（Follower Velocity）是区分力最强的真实数据指标。**上线前30天的关注者日增速 Cohen's d = {d["fol_30d"]["d"]}，达到「大效应」水平。爆款游戏的中位数日增 **{int(d["fol_30d"]["s_median"])}** 人/天，对照组仅 **{int(d["fol_30d"]["c_median"])}** 人/天，差距高达 **{d["fol_30d"]["s_median"]/d["fol_30d"]["c_median"]:.1f} 倍**。

"""
    md += """退一步看，这个发现其实在说一件很直觉的事：**增速比绝对数量更重要。**一个有20万愿望单但增速为0的游戏，远不如一个5万愿望单但每天涨1000的游戏有潜力。增速捕捉的是「势头」——活跃的、正在增长的兴趣，而不是冷冰冰的历史积累。

### 那些「平平无奇」的指标

有几个你可能以为很重要的指标，其实完全没用：

| 指标 | Cohen's d | 意味着什么 |
|------|-----------|-----------|
| 售价 | 0.09 | 免费（漫威争锋）和 $60（黑神话）都能爆 |
| 开发者历史营收 | 0.01 | 新工作室和老工作室一样可能出爆款 |
| 上线前挂页天数 | 0.08 | 挂半年和挂两年都行 |
| 标签数量 | 0.25 | 标签多少和成功无关 |

特别值得一提的是**开发者历史（d=0.01）**——这是整个研究中最令人意外的发现之一。它几乎等于零，完美印证了「潜力爆款」的本质：它们就是来自意想不到的地方。致命公司的开发者 Zeekerss 之前毫无作品；小丑牌的 LocalThunk 也是第一次做游戏。**你无法从开发者的简历预测爆款。**

### 爆款游戏的增长轨迹长什么样？

[图表：3款低调爆款的关注者增长轨迹 vs 对照组中位数]

即使不看数字，轨迹的形态差异也一目了然：爆款游戏的曲线明显更陡，而且越接近上线越陡。这引出了一个关键问题——

> **既然增速这么重要，我们有没有可能不等到临门一脚，而是更早、更前瞻地发现这些「潜力股」呢？**

---

## 第二章：车前灯——提前识别爆款

后视镜的问题在于——它是后视镜。「上线前30天增速」这个指标，你得等到游戏上线后才能算出来。没人会倒计时着去做判断。那么如果我们想「前向」预测，选什么速度指标最好？

### 前向指标：Peak Week Velocity 胜出

我们测试了两个前向可观测的指标：

- **Peak Week Velocity（巅峰周增速）**：一个游戏在上线前，关注者7日滚动均增最高的一周。通常对应一次重要事件——比如预告片发布、试玩节参展、主播带火。
- **Avg Daily Velocity（日常增速）**：所有观测点的7日滚动均增的中位数，反映游戏的「日常」吸粉能力。

"""
    md += f"""结论清晰：**巅峰周增速是更好的前向信号。**这很符合直觉——一款游戏的「天花板时刻」（最火的那一周）最能代表它的潜力上限，而日均增速会被漫长的冷淡期拉低。

[图表：巅峰周增速分布 - 爆款 vs 对照组]

爆款中位数 {int(d["peak_summary"]["s_median"])} 人/天，对照组 {int(d["peak_summary"]["c_median"])} 人/天，差距 {d["peak_summary"]["ratio"]:.1f} 倍。

### 这些巅峰时刻发生在什么时候？

[图表：选定游戏的7日滚动均增轨迹，标注巅峰周位置]

有趣的是，巅峰周可能出现在上线前很久（比如 Manor Lords 在 Demo 发布后），也可能出现在上线前几周（比如 Schedule I 在口碑发酵后）。关键不是*什么时候*出现巅峰，而是*巅峰有多高*。

### 增速差距从什么时候开始存在？

[图表：不同时间窗口的关注者中位增速对比]

> **关键发现：增速差距从第一天就存在。**爆款游戏从 Steam 页面创建之初，关注者日增速就是对照组的 5-8 倍。差距在上线前30天扩大到 8.5 倍，但即使在上线前一年就已经是 5.5 倍。换句话说——**你不需要等到临门一脚，越早看到信号，判断力越强。**

### 门槛定多少？Hit Rate 和 Catch Rate 的取舍

如果我们用巅峰周增速作为筛选标准，接下来最重要的问题是：**门槛定在多少？**

这里存在一个永恒的取舍：

- **命中率（Hit Rate）**：「我标记的游戏里，有多少是真正的爆款？」门槛越高，命中率越好，但你会漏掉更多。
- **覆盖率（Catch Rate）**：「所有爆款里，有多少被我标记到了？」门槛越低，覆盖越多，但你也标记了一堆非爆款。

鱼和熊掌不可兼得。最优门槛取决于你的使用场景。

[图表：命中率 vs 覆盖率随门槛变化曲线]

### 四种策略，各取所需

"""
    for t in d["tiers"]:
        md += f"- **{t['cn_label']}**（≥{t['threshold']}/天）：关注 {t['radar']} 款，命中率 {t['precision']:.0%}，覆盖率 {t['recall']:.0%}\n"

    md += """
[交互表格：每个策略下的命中/遗漏/误判/完整关注列表]

> **推荐：甜点策略（≥845/天）**——关注50款游戏，每3个中有1个是真正的爆款，覆盖60%的已知爆款。如果你是做投资/合作的，这是投入产出比最好的平衡点。

---

## 第三章：其他发现

### 爆款的标签画像：什么类型更容易爆？

[图表：标签在爆款组中的出现频率差异]

一个清晰的画像浮现出来：爆款倾向于**多人社交体验**——在线合作、角色自定义、第三人称射击、提取射击、生存。而单人、休闲、模拟类的产品相对较少出现在爆款名单中（尽管小丑牌和 Manor Lords 是显著的例外）。

### 所谓的「误判」，真的是误判吗？

被我们的筛选标记为「高增速」但不在爆款名单里的对照组游戏，包括：战地6、对马岛之魂、Split Fiction、潜行者2、战锤40K、三角洲部队、最终幻想7 重生、沙丘：觉醒、杀戮尖塔2…

发现了吗？这些所谓的「误判」几乎全是**商业上非常成功的大作**——它们没进爆款名单，只是因为我们的名单针对的是「超预期惊喜」，而这些大作的成功本就在意料之中。

所以筛选器实际上捕捉的是**所有商业表现优秀的游戏**，不仅仅是惊喜爆款——这反而让它作为投资/合作的雷达更有价值。

### 约 20% 的爆款在上线前完全隐身

有一个重要的局限需要坦诚面对：并非所有爆款都能被提前识别。

| 游戏 | 上线前日增 | 为什么检测不到 | 首月营收 |
|------|-----------|-------------|---------|
| 致命公司 | 47/天 | 仅在Steam上挂了46天，初始仅340关注者 | $146M |
| 恶魔轮盘 | 216/天 | 微型独立游戏，极短的上线前窗口 | $17M |
| 小丑牌 | 214/天 | 上线前低调，靠口碑传播爆发 | $78M |
| Megabonk | 84/天 | 近乎零预热信号 | $32M |

这些是典型的「发售日爆发型」——它们的病毒传播发生在**上线当天或之后**（主播发现、社交媒体引爆），任何上线前的指标都无法捕捉它们。**这意味着前向筛选大约能覆盖 80% 的爆款，剩下 20% 需要通过上线后的实时监控来发现。**

### 高增速是必要条件，但不是充分条件

| | 好游戏 | 差体验 |
|---|---|---|
| **高增速** | 爆款（黑神话、帕鲁、Manor Lords） | 失望（Nightingale、ASKA、Avowed） |
| **低增速** | 黑马（致命公司、Content Warning、小丑牌） | 沉寂（大多数游戏） |

关注者增速能告诉你「这款游戏有人看」，但不能告诉你「这款游戏好不好玩」。高增速 + 差体验的组合会产生「高期待落空」型的失望案例（如 Nightingale，上线前731/天增速，首月仅$4M）。

### 关注者 vs 营收：Ground Truth 的力量

[图表：上线时关注者数 vs 首月营收散点图]

### 「深口袋」假说被否定

你可能会想：是不是那些爆款只是因为砸了更多钱做营销？答案恰恰相反：**63% 的爆款是自发行**（开发者自己当发行商），对照组只有 49%。大发行商（Xbox、EA、Bandai Namco、Square Enix、SEGA）反而更多出现在对照组中。所谓的「深口袋」假说，在数据面前完全站不住。

---

## 研究局限性

这份研究中，仍有较多的局限性：

1. **Gamalytic 追踪缺口**：有3款爆款（PEAK、Dark and Darker、Content Warning）因 Steam 页面创建极晚或 Gamalytic 开始追踪较晚，没有足够的上线前关注者数据（<7天）。
2. **前向评估样本的局限**：策略评估基于 Gamalytic 有上线前追踪的 ~3,800 款游戏样本。这些游戏偏向有一定关注度的产品（followers ≥ 500），不完全代表 Steam 全量 6 万+游戏。
3. **无因果推断**：高关注者增速可能是病毒传播的*结果*（TikTok 带火 → 关注者暴涨），而非*原因*。不存在"因为这个产品Follower增长很快所以必然大卖"这样的因果关系
4. **营收/销量仍为估算值**：只有关注者数据经 Gamalytic 确认为 Steam 真实数据，营收和销量数据均为算法估算。

---

*Steam 潜力爆款识别研究 · 数据来源：Gamalytic API · 2026年3月*
"""
    return md


# ── Main ───────────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    af, fwd, bd, success_names, name_lookup = load_all_data()

    print("Extracting separation power...")
    sep_power = extract_separation_power(af)

    print("Extracting success games table...")
    success_games = extract_success_games_table(af, bd, success_names)

    print("Extracting trajectories...")
    # Pick 3 low-key success games for showcase
    showcase_ids = [
        "3164500",  # Schedule I
        "3241660",  # R.E.P.O.
        "2379780",  # Balatro
    ]
    trajectories = extract_trajectories(fwd, showcase_ids, name_lookup)
    control_envelope = extract_control_envelope(fwd)

    print("Extracting peak week data...")
    peak_data = extract_peak_week_data(fwd, name_lookup)
    s_peaks = [g["peak_week"] for g in peak_data if g["group"] == "success"]
    c_peaks = [g["peak_week"] for g in peak_data if g["group"] == "control"]
    peak_summary = {
        "s_median": round(float(np.median(s_peaks)), 0) if s_peaks else 0,
        "c_median": round(float(np.median(c_peaks)), 0) if c_peaks else 0,
        "ratio": round(float(np.median(s_peaks)) / float(np.median(c_peaks)), 1) if c_peaks and np.median(c_peaks) > 0 else 0,
    }

    print("Extracting peak velocity from ALL history files (backdrop sample)...")
    success_ids = set(success_names.keys())
    peak_data_full = extract_peak_week_from_histories(success_ids, name_lookup)

    print("Extracting threshold sweep (full backdrop, avg floor=20)...")
    threshold_sweep = extract_threshold_sweep(peak_data_full, avg_floor=20)

    print("Extracting strategy tiers (full backdrop)...")
    tiers = extract_strategy_tiers(peak_data_full, name_lookup, success_names)

    print("Extracting tag analysis...")
    tags = extract_tag_analysis(af)

    print("Extracting velocity windows...")
    velocity_windows = extract_velocity_windows(fwd)

    print("Extracting scatter data...")
    scatter = extract_scatter_data(af, name_lookup, peak_data_full)

    print("Extracting peak annotated trajectories...")
    # Mix of early-peak and late-peak games for diversity (Palworld removed per v4)
    peak_annotated_ids = [
        "2767030",  # Marvel Rivals (peak ~T-245, early) — green
        "1808500",  # ARC Raiders (peak ~T-178, very early) — blue
        "1363080",  # Manor Lords (peak ~T-7, late) — amber
        "3164500",  # Schedule I (peak ~T-12, late) — red
    ]
    peak_annotated = extract_trajectory_with_peak(fwd, peak_annotated_ids, name_lookup, peak_data)

    print("Extracting detection lead time...")
    detection_leadtime = extract_detection_leadtime(fwd, success_names)

    print("Extracting detection lead time for $50M+ games (min_n=30)...")
    detection_50m = extract_detection_leadtime_50m(peak_data_full, name_lookup=name_lookup)

    print("Computing peak week Cohen's d (full backdrop)...")
    peak_d_comparison = extract_peak_cohens_d(peak_data, peak_data_full)

    print("Computing strategy quality metrics (Top100 against full backdrop)...")
    strategy_quality = extract_strategy_quality(peak_data_full, bd=bd)

    # Find follower 30d velocity specifically for highlight text
    fol_30d = next((f for f in sep_power if f["col"] == "follower_velocity_30d"), sep_power[0])

    # Backdrop stats for the "前向预判" subsection
    total_backdrop = len(bd)
    n_with_prelaunch = len(peak_data_full)
    backdrop_stats = {
        "total_games": int(total_backdrop),
        "sample_size": n_with_prelaunch,
        "n_success": int((pd.DataFrame(peak_data_full)["group"] == "success").sum()),
        "n_backdrop": n_with_prelaunch - int((pd.DataFrame(peak_data_full)["group"] == "success").sum()),
    }

    # Assemble report data
    report_data = {
        "success_games": success_games,
        "trajectories": trajectories,
        "control_envelope": control_envelope,
        "peak_distribution_full": peak_data_full,
        "peak_annotated": peak_annotated,
        "peak_d_comparison": peak_d_comparison,
        "threshold_sweep": threshold_sweep,
        "tiers": tiers,
        "velocity_windows": velocity_windows,
        "detection_leadtime": detection_leadtime,
        "detection_50m": detection_50m,
        "backdrop_stats": backdrop_stats,
        "strategy_quality": strategy_quality,
        # Legacy: kept for generate_md() compatibility
        "fol_30d": fol_30d,
        "peak_summary": peak_summary,
    }

    print("Generating HTML report...")
    html = generate_html(report_data)
    html_path = OUTPUT_DIR / "report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  → {html_path} ({len(html):,} bytes)")

    print("Generating MD text...")
    md = generate_md(report_data)
    md_path = OUTPUT_DIR / "report_text.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  → {md_path} ({len(md):,} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    run()
