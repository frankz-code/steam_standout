"""PID signal experiment: test Proportional / Integral / Derivative signals
for predicting Steam game commercial success (>=$50M revenue).

Standalone experiment — does not modify existing pipeline code.
Reads from data/histories/*.json (same data as 10_generate_report.py).

Signal definitions (signal = daily follower increment):
  P: peak_week (7d rolling max) + avg_daily (7d rolling median)  — existing baseline
  I: cumulative followers at last pre-launch observation          — "installed base"
  D: OLS slope of 7d rolling velocity over last 60 days          — "acceleration"
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
HISTORIES_DIR = DATA_DIR / "histories"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"

REVENUE_THRESHOLD = 50_000_000  # $50M


def load_name_lookup():
    """Load game name lookup from backdrop_games.parquet and success_games.json."""
    names = {}
    backdrop = DATA_DIR / "backdrop_games.parquet"
    if backdrop.exists():
        df = pd.read_parquet(backdrop, columns=["steamId", "name"])
        for _, row in df.iterrows():
            names[str(row["steamId"])] = row["name"]
    if SUCCESS_FILE.exists():
        with open(SUCCESS_FILE) as f:
            for g in json.load(f):
                names[str(g["steamId"])] = g["name"]
    return names


def extract_pid_signals(success_ids):
    """Extract P, I, D signals from all history JSON files."""
    name_lookup = load_name_lookup()
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

        # Pre-launch only
        pre = df[df["days_to_launch"] > 0].copy().sort_values("date")
        if len(pre) < 2:
            skipped += 1
            continue

        # ── I signal: cumulative followers at last pre-launch observation ──
        i_signal = float(pre.iloc[-1]["followers"])

        # ── P signals: peak_week and avg_daily (replicate existing logic) ──
        pre["fol_diff"] = pre["followers"].diff()
        pre["date_diff"] = pre["date"].diff().dt.total_seconds() / 86400
        pre["daily_rate"] = pre["fol_diff"] / pre["date_diff"]

        # Onset artifact: null first rate if initial followers > 100
        if pre.iloc[0]["followers"] > 100 and len(pre) > 1:
            pre.iloc[0, pre.columns.get_loc("daily_rate")] = np.nan

        rates = pre["daily_rate"].dropna()
        if len(rates) < 2:
            skipped += 1
            continue

        # 7-day rolling mean of daily rate
        if len(rates) >= 7:
            rolling = rates.rolling(7, min_periods=4).mean().dropna()
        else:
            rolling = rates

        if rolling.empty:
            skipped += 1
            continue

        peak_week = float(rolling.max())
        avg_daily = float(rolling.median())

        # ── D signal: OLS slope of rolling velocity over last 30 days ──
        d_signal = np.nan
        if len(rolling) >= 3:
            # Map rolling values back to days_to_launch
            rolling_with_dtl = pd.DataFrame({
                "velocity": rolling,
                "days_to_launch": pre.loc[rolling.index, "days_to_launch"],
            })
            r60 = rolling_with_dtl[rolling_with_dtl["days_to_launch"] <= 30].dropna()

            if len(r60) >= 3:
                # x = -days_to_launch so it increases toward launch
                x = -r60["days_to_launch"].values.astype(float)
                y = r60["velocity"].values
                x_mean, y_mean = x.mean(), y.mean()
                denom = np.sum((x - x_mean) ** 2)
                if denom > 0:
                    d_signal = float(np.sum((x - x_mean) * (y - y_mean)) / denom)

        # Revenue
        revenue = data.get("revenue") or data.get("totalRevenue") or 0
        revenue = int(revenue) if revenue else 0

        group = "success" if sid in success_ids else "backdrop"

        games.append({
            "steamId": sid,
            "name": name_lookup.get(sid, data.get("name", sid)),
            "group": group,
            "peak_week": round(peak_week, 1),
            "avg_daily": round(avg_daily, 1),
            "i_signal": round(i_signal, 0),
            "d_signal": round(d_signal, 4) if not np.isnan(d_signal) else np.nan,
            "revenue": revenue,
        })

    print(f"Extracted PID signals for {len(games)} games (skipped {skipped})")
    return pd.DataFrame(games)


def evaluate_signal(df, signal_col, percentiles):
    """Sweep percentile thresholds for a single signal."""
    total_hits = (df["revenue"] >= REVENUE_THRESHOLD).sum()
    base_rate = total_hits / len(df) if len(df) > 0 else 0
    results = []

    valid = df[df[signal_col].notna()]
    for p in percentiles:
        t = np.percentile(valid[signal_col], p)
        selected = valid[valid[signal_col] >= t]
        watch = len(selected)
        hits = (selected["revenue"] >= REVENUE_THRESHOLD).sum()
        hit_rate = hits / watch if watch > 0 else 0
        catch = hits / total_hits if total_hits > 0 else 0
        lift = hit_rate / base_rate if base_rate > 0 else 0
        results.append({
            "pctl": p, "threshold": t,
            "watch": watch, "hits": hits,
            "hit_rate": hit_rate, "catch": catch, "lift": lift,
        })
    return results


def evaluate_combo(df, signals_and_thresholds, label):
    """Evaluate AND combination of multiple signal thresholds."""
    total_hits = (df["revenue"] >= REVENUE_THRESHOLD).sum()
    base_rate = total_hits / len(df) if len(df) > 0 else 0

    mask = pd.Series(True, index=df.index)
    for col, t in signals_and_thresholds:
        mask &= df[col] >= t
    selected = df[mask]
    watch = len(selected)
    hits = (selected["revenue"] >= REVENUE_THRESHOLD).sum()
    hit_rate = hits / watch if watch > 0 else 0
    catch = hits / total_hits if total_hits > 0 else 0
    lift = hit_rate / base_rate if base_rate > 0 else 0
    return {
        "signal": label, "watch": watch, "hits": hits,
        "hit_rate": hit_rate, "catch": catch, "lift": lift,
    }


# ── Composite scoring approach ──────────────────────────────────────

def compute_composite_scores(df, signal_cols, weights):
    """Convert signals to percentile ranks, return weighted composite score."""
    rank_cols = {}
    for col in signal_cols:
        valid_mask = df[col].notna()
        rank_cols[col] = df[col].rank(pct=True) * 100
        rank_cols[col][~valid_mask] = np.nan
    score = sum(rank_cols[col] * w for col, w in zip(signal_cols, weights))
    return score


def build_pr_curve(df, score_col, n_points=200):
    """Build precision-recall curve by sweeping threshold on a score column."""
    valid = df[df[score_col].notna()].copy()
    total_hits = (valid["revenue"] >= REVENUE_THRESHOLD).sum()
    base_rate = total_hits / len(valid) if len(valid) > 0 else 0
    if total_hits == 0:
        return []

    thresholds = np.linspace(valid[score_col].min(), valid[score_col].max(), n_points)
    curve = []
    for t in thresholds:
        sel = valid[valid[score_col] >= t]
        watch = len(sel)
        if watch == 0:
            continue
        hits = (sel["revenue"] >= REVENUE_THRESHOLD).sum()
        hit_rate = hits / watch
        catch = hits / total_hits
        lift = hit_rate / base_rate if base_rate > 0 else 0
        curve.append({"threshold": t, "watch": watch, "hits": hits,
                       "hit_rate": hit_rate, "catch": catch, "lift": lift})
    return curve


def find_at_catch(curve, target_catch):
    """Find operating point closest to target catch rate (from above)."""
    best = None
    for pt in curve:
        if pt["catch"] >= target_catch:
            if best is None or pt["watch"] < best["watch"]:
                best = pt
    return best


def print_row(r):
    print(f"  {r['signal']:<20} {r['watch']:>6} {r['hits']:>5}"
          f" {r['hit_rate']:>7.1%} {r['catch']:>6.1%} {r['lift']:>5.1f}x")


def main():
    # Load success game IDs
    with open(SUCCESS_FILE) as f:
        success_games = json.load(f)
    success_ids = set(str(g["steamId"]) for g in success_games)

    # Extract all signals
    df = extract_pid_signals(success_ids)

    # Filter to games with revenue data
    df_valid = df[df["revenue"] > 0].copy()
    total = len(df_valid)
    total_hits = (df_valid["revenue"] >= REVENUE_THRESHOLD).sum()
    base_rate = total_hits / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"PID Signal Experiment — ${REVENUE_THRESHOLD/1e6:.0f}M Revenue Prediction")
    print(f"{'='*70}")
    print(f"Games with revenue data: {total}")
    print(f"Games >= ${REVENUE_THRESHOLD/1e6:.0f}M: {total_hits}")
    print(f"Base rate: {base_rate:.2%}")

    # Signal coverage & distribution
    print(f"\n{'─'*70}")
    print("Signal Distributions")
    print(f"{'─'*70}")
    for col, label in [("peak_week", "P: peak_week"), ("avg_daily", "P: avg_daily"),
                        ("i_signal", "I: cumul_followers"), ("d_signal", "D: acceleration")]:
        valid = df_valid[col].dropna()
        print(f"  {label:<25} n={len(valid):>5}  "
              f"p50={np.percentile(valid,50):>10.1f}  "
              f"p90={np.percentile(valid,90):>10.1f}  "
              f"p99={np.percentile(valid,99):>10.1f}")

    # ── Single signal sweeps ──
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]

    print(f"\n{'─'*70}")
    print("Single Signal Sweep")
    print(f"{'─'*70}")

    for col, label in [("peak_week", "P: peak_week"), ("avg_daily", "P: avg_daily"),
                        ("i_signal", "I: cumul_followers"), ("d_signal", "D: acceleration")]:
        valid = df_valid[df_valid[col].notna()]
        if len(valid) < 10:
            print(f"\n{label}: insufficient data ({len(valid)} games)")
            continue

        results = evaluate_signal(valid, col, percentiles)
        print(f"\n  {label} (n={len(valid)}):")
        print(f"  {'Pctl':>5} {'Thresh':>12} {'Watch':>6} {'Hits':>5} {'HitRate':>8} {'Catch':>7} {'Lift':>5}")
        for r in results:
            print(f"  {r['pctl']:>4}% {r['threshold']:>12.1f} {r['watch']:>6} {r['hits']:>5}"
                  f" {r['hit_rate']:>7.1%} {r['catch']:>6.1%} {r['lift']:>5.1f}x")

    # ── Combination evaluations ──
    print(f"\n{'─'*70}")
    print("Combination Signals (AND logic, compared at matched percentiles)")
    print(f"{'─'*70}")

    # Require all signals valid for fair comparison
    df_combo = df_valid[df_valid["d_signal"].notna()].copy()
    combo_total = len(df_combo)
    combo_hits = (df_combo["revenue"] >= REVENUE_THRESHOLD).sum()
    print(f"Games with all signals: {combo_total} (${REVENUE_THRESHOLD/1e6:.0f}M hits: {combo_hits})")

    combo_pctls = [70, 80, 90, 95]

    print(f"\n  {'Combo':<20} {'Watch':>6} {'Hits':>5} {'HitRate':>8} {'Catch':>7} {'Lift':>5}")
    print(f"  {'─'*57}")

    for p in combo_pctls:
        pk_t = np.percentile(df_combo["peak_week"], p)
        avg_t = np.percentile(df_combo["avg_daily"], p)
        i_t = np.percentile(df_combo["i_signal"], p)
        d_t = np.percentile(df_combo["d_signal"], p)

        print(f"  --- @{p}th percentile ---")

        # P only (baseline: peak + avg)
        print_row(evaluate_combo(df_combo, [("peak_week", pk_t), ("avg_daily", avg_t)], "P (baseline)"))
        # I only
        print_row(evaluate_combo(df_combo, [("i_signal", i_t)], "I only"))
        # D only
        print_row(evaluate_combo(df_combo, [("d_signal", d_t)], "D only"))
        # P + I
        print_row(evaluate_combo(df_combo, [("peak_week", pk_t), ("avg_daily", avg_t), ("i_signal", i_t)], "P + I"))
        # P + D
        print_row(evaluate_combo(df_combo, [("peak_week", pk_t), ("avg_daily", avg_t), ("d_signal", d_t)], "P + D"))
        # P + I + D
        print_row(evaluate_combo(df_combo,
                  [("peak_week", pk_t), ("avg_daily", avg_t), ("i_signal", i_t), ("d_signal", d_t)], "P + I + D"))
        print()

    # ══════════════════════════════════════════════════════════════════
    # Part 2: Composite Scoring (elastic alternative to AND)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Part 2: Composite Scoring (weighted percentile ranks)")
    print(f"{'='*70}")

    # Work with games that have all signals
    df_score = df_combo.copy()

    # Define scoring schemes to compare
    schemes = [
        ("P only",   ["peak_week", "avg_daily"],                       [0.5, 0.5]),
        ("P+I",      ["peak_week", "avg_daily", "i_signal"],           [0.35, 0.35, 0.3]),
        ("P+D",      ["peak_week", "avg_daily", "d_signal"],           [0.35, 0.35, 0.3]),
        ("P+I+D",    ["peak_week", "avg_daily", "i_signal", "d_signal"], [0.3, 0.3, 0.2, 0.2]),
        ("I only",   ["i_signal"],                                      [1.0]),
        ("D only",   ["d_signal"],                                      [1.0]),
    ]

    # Compute composite scores and PR curves
    curves = {}
    for name, cols, weights in schemes:
        score_col = f"score_{name.replace('+','_').replace(' ','_')}"
        df_score[score_col] = compute_composite_scores(df_score, cols, weights)
        curves[name] = build_pr_curve(df_score, score_col)

    # ── Fixed-catch comparison ──
    print(f"\n{'─'*70}")
    print("Fixed Catch Rate Comparison")
    print("(At equal catch, fewer Watch = better signal)")
    print(f"{'─'*70}")

    target_catches = [0.90, 0.80, 0.60, 0.40]

    for target in target_catches:
        print(f"\n  Target catch = {target:.0%}:")
        print(f"  {'Scheme':<15} {'Watch':>6} {'Hits':>5} {'HitRate':>8} {'Catch':>7} {'Lift':>5}")
        print(f"  {'─'*50}")

        for name, _, _ in schemes:
            pt = find_at_catch(curves[name], target)
            if pt:
                print(f"  {name:<15} {pt['watch']:>6} {pt['hits']:>5}"
                      f" {pt['hit_rate']:>7.1%} {pt['catch']:>6.1%} {pt['lift']:>5.1f}x")
            else:
                print(f"  {name:<15}    --- cannot reach {target:.0%} catch ---")

    # ── PR curve summary (AUC approximation) ──
    print(f"\n{'─'*70}")
    print("PR Curve Quality (AUC = area under hit_rate vs catch curve)")
    print(f"{'─'*70}")

    for name, _, _ in schemes:
        curve = curves[name]
        if len(curve) < 2:
            print(f"  {name:<15}  insufficient data")
            continue
        # Sort by catch ascending for AUC
        pts = sorted(curve, key=lambda x: x["catch"])
        catches = [p["catch"] for p in pts]
        hit_rates = [p["hit_rate"] for p in pts]
        # Trapezoidal AUC
        auc = np.trapezoid(hit_rates, catches)
        print(f"  {name:<15}  AUC = {auc:.4f}")

    # ── Plot PR curves ──
    print(f"\n{'─'*70}")
    print("Generating PR curve plot...")
    print(f"{'─'*70}")

    plt.rcParams.update({
        "font.family": ["PingFang SC", "Heiti SC", "Microsoft YaHei", "sans-serif"],
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "P only": "#4A90D9",
        "P+I": "#E74C3C",
        "P+D": "#95A5A6",
        "P+I+D": "#F39C12",
        "I only": "#2ECC71",
        "D only": "#BDC3C7",
    }
    linewidths = {
        "P only": 2.5,
        "P+I": 3.0,
        "P+D": 1.5,
        "P+I+D": 2.0,
        "I only": 1.5,
        "D only": 1.0,
    }

    for name, _, _ in schemes:
        curve = curves[name]
        if len(curve) < 2:
            continue
        pts = sorted(curve, key=lambda x: x["catch"])
        catches = [p["catch"] for p in pts]
        hit_rates = [p["hit_rate"] for p in pts]
        auc = np.trapezoid(hit_rates, catches)
        ax.plot(catches, hit_rates,
                color=colors.get(name, "#666"),
                linewidth=linewidths.get(name, 1.5),
                label=f"{name}  (AUC={auc:.3f})",
                alpha=0.9)

    # Base rate reference line
    base = combo_hits / combo_total
    ax.axhline(y=base, color="#CCC", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.02, base + 0.01, f"base rate {base:.1%}", color="#999", fontsize=9)

    ax.set_xlabel("Catch Rate (proportion of $50M+ games captured)", fontsize=13)
    ax.set_ylabel("Hit Rate (precision among selected games)", fontsize=13)
    ax.set_title("PID Signal Comparison: Precision–Recall Curves\n$50M+ Revenue Prediction on 3,298 Steam Games", fontsize=14)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    plot_path = PROJECT_ROOT / "output" / "exp_pid_pr_curves.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"PR curve saved to {plot_path}")

    # ══════════════════════════════════════════════════════════════════
    # Part 3: P+I vs P only — concrete game-level diff
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Part 3: P+I vs P only — Which games differ?")
    print(f"{'='*70}")

    p_col = "score_P_only"
    pi_col = "score_P_I"

    for target in [0.80, 0.60]:
        pt_p = find_at_catch(curves["P only"], target)
        pt_pi = find_at_catch(curves["P+I"], target)
        if not pt_p or not pt_pi:
            continue

        sel_p = set(df_score[df_score[p_col] >= pt_p["threshold"]].index)
        sel_pi = set(df_score[df_score[pi_col] >= pt_pi["threshold"]].index)

        only_p = sel_p - sel_pi   # P selects but P+I doesn't
        only_pi = sel_pi - sel_p  # P+I selects but P doesn't
        both = sel_p & sel_pi

        print(f"\n  ── At catch ≈ {target:.0%} ──")
        print(f"  P only: {len(sel_p)} games (hit rate {pt_p['hit_rate']:.1%})")
        print(f"  P+I:    {len(sel_pi)} games (hit rate {pt_pi['hit_rate']:.1%})")
        print(f"  Overlap: {len(both)}  |  P only exclusive: {len(only_p)}  |  P+I exclusive: {len(only_pi)}")

        # Games P selects but P+I drops — are these false positives?
        if only_p:
            dropped = df_score.loc[list(only_p)].sort_values("revenue", ascending=False)
            n_hit = (dropped["revenue"] >= REVENUE_THRESHOLD).sum()
            n_miss = len(dropped) - n_hit
            print(f"\n  P+I drops {len(only_p)} games that P kept ({n_hit} hits, {n_miss} non-hits):")
            print(f"  {'Name':<40} {'Revenue':>12} {'PeakWk':>8} {'AvgDly':>8} {'Followers':>10} {'$50M?':>6}")
            for _, r in dropped.head(20).iterrows():
                is_hit = "HIT" if r["revenue"] >= REVENUE_THRESHOLD else ""
                print(f"  {str(r['name'])[:39]:<40} ${r['revenue']/1e6:>9.1f}M"
                      f" {r['peak_week']:>8.0f} {r['avg_daily']:>8.1f} {r['i_signal']:>10.0f} {is_hit:>6}")

        # Games P+I selects but P doesn't — what did I signal add?
        if only_pi:
            added = df_score.loc[list(only_pi)].sort_values("revenue", ascending=False)
            n_hit = (added["revenue"] >= REVENUE_THRESHOLD).sum()
            n_miss = len(added) - n_hit
            print(f"\n  P+I adds {len(only_pi)} games that P missed ({n_hit} hits, {n_miss} non-hits):")
            print(f"  {'Name':<40} {'Revenue':>12} {'PeakWk':>8} {'AvgDly':>8} {'Followers':>10} {'$50M?':>6}")
            for _, r in added.head(20).iterrows():
                is_hit = "HIT" if r["revenue"] >= REVENUE_THRESHOLD else ""
                print(f"  {str(r['name'])[:39]:<40} ${r['revenue']/1e6:>9.1f}M"
                      f" {r['peak_week']:>8.0f} {r['avg_daily']:>8.1f} {r['i_signal']:>10.0f} {is_hit:>6}")

    # ══════════════════════════════════════════════════════════════════
    # Part 4: Re-evaluate using Success Set (30 curated games) as truth
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Part 4: Success Set (30 curated games) as hit definition")
    print(f"{'='*70}")

    df_s = df_score.copy()
    df_s["is_hit"] = df_s["group"] == "success"
    total_s = len(df_s)
    total_hits_s = df_s["is_hit"].sum()
    base_rate_s = total_hits_s / total_s
    print(f"Games: {total_s}  |  Success set in data: {total_hits_s}  |  Base rate: {base_rate_s:.2%}")

    # Build PR curves with success-set truth
    def build_pr_curve_custom(df, score_col, hit_col="is_hit"):
        valid = df[df[score_col].notna()].copy()
        total_h = valid[hit_col].sum()
        base = total_h / len(valid) if len(valid) > 0 else 0
        if total_h == 0:
            return []
        thresholds = np.linspace(valid[score_col].min(), valid[score_col].max(), 200)
        curve = []
        for t in thresholds:
            sel = valid[valid[score_col] >= t]
            w = len(sel)
            if w == 0:
                continue
            h = sel[hit_col].sum()
            hr = h / w
            c = h / total_h
            curve.append({"threshold": t, "watch": w, "hits": int(h),
                          "hit_rate": hr, "catch": c, "lift": hr / base if base > 0 else 0})
        return curve

    curves_s = {}
    for name, cols, weights in schemes:
        score_col = f"score_{name.replace('+','_').replace(' ','_')}"
        curves_s[name] = build_pr_curve_custom(df_s, score_col)

    # Fixed catch comparison
    print(f"\n{'─'*70}")
    print("Fixed Catch Rate Comparison (hit = success set)")
    print(f"{'─'*70}")

    for target in [0.80, 0.60, 0.40]:
        print(f"\n  Target catch = {target:.0%}:")
        print(f"  {'Scheme':<15} {'Watch':>6} {'Hits':>5} {'HitRate':>8} {'Catch':>7} {'Lift':>5}")
        print(f"  {'─'*50}")
        for name, _, _ in schemes:
            pt = find_at_catch(curves_s[name], target)
            if pt:
                print(f"  {name:<15} {pt['watch']:>6} {pt['hits']:>5}"
                      f" {pt['hit_rate']:>7.1%} {pt['catch']:>6.1%} {pt['lift']:>5.1f}x")
            else:
                print(f"  {name:<15}    --- cannot reach {target:.0%} catch ---")

    # AUC
    print(f"\n{'─'*70}")
    print("PR Curve AUC (hit = success set)")
    print(f"{'─'*70}")
    for name, _, _ in schemes:
        curve = curves_s[name]
        if len(curve) < 2:
            continue
        pts = sorted(curve, key=lambda x: x["catch"])
        auc = np.trapezoid([p["hit_rate"] for p in pts], [p["catch"] for p in pts])
        print(f"  {name:<15}  AUC = {auc:.4f}")

    # Game-level diff at catch=60%
    print(f"\n{'─'*70}")
    print("P+I vs P only — Game-level diff (hit = success set)")
    print(f"{'─'*70}")

    p_col = "score_P_only"
    pi_col = "score_P_I"

    for target in [0.80, 0.60]:
        pt_p = find_at_catch(curves_s["P only"], target)
        pt_pi = find_at_catch(curves_s["P+I"], target)
        if not pt_p or not pt_pi:
            continue

        sel_p = set(df_s[df_s[p_col] >= pt_p["threshold"]].index)
        sel_pi = set(df_s[df_s[pi_col] >= pt_pi["threshold"]].index)

        only_p = sel_p - sel_pi
        only_pi = sel_pi - sel_p

        print(f"\n  ── At catch ≈ {target:.0%} ──")
        print(f"  P only: {len(sel_p)} games (hit rate {pt_p['hit_rate']:.1%})")
        print(f"  P+I:    {len(sel_pi)} games (hit rate {pt_pi['hit_rate']:.1%})")

        if only_p:
            dropped = df_s.loc[list(only_p)].sort_values("revenue", ascending=False)
            n_hit = dropped["is_hit"].sum()
            print(f"\n  P+I drops {len(only_p)} games ({n_hit} success, {len(only_p)-n_hit} non-success):")
            print(f"  {'Name':<40} {'Revenue':>12} {'PeakWk':>8} {'AvgDly':>8} {'Foll':>10} {'Succ?':>6}")
            for _, r in dropped.head(25).iterrows():
                tag = "YES" if r["is_hit"] else ""
                print(f"  {str(r['name'])[:39]:<40} ${r['revenue']/1e6:>9.1f}M"
                      f" {r['peak_week']:>8.0f} {r['avg_daily']:>8.1f} {r['i_signal']:>10.0f} {tag:>6}")

        if only_pi:
            added = df_s.loc[list(only_pi)].sort_values("revenue", ascending=False)
            n_hit = added["is_hit"].sum()
            print(f"\n  P+I adds {len(only_pi)} games ({n_hit} success, {len(only_pi)-n_hit} non-success):")
            print(f"  {'Name':<40} {'Revenue':>12} {'PeakWk':>8} {'AvgDly':>8} {'Foll':>10} {'Succ?':>6}")
            for _, r in added.head(25).iterrows():
                tag = "YES" if r["is_hit"] else ""
                print(f"  {str(r['name'])[:39]:<40} ${r['revenue']/1e6:>9.1f}M"
                      f" {r['peak_week']:>8.0f} {r['avg_daily']:>8.1f} {r['i_signal']:>10.0f} {tag:>6}")

    # Save detailed data
    output_path = DATA_DIR / "exp_pid_signals.csv"
    df_score.to_csv(output_path, index=False)
    print(f"\nSignal data saved to {output_path}")


if __name__ == "__main__":
    main()
