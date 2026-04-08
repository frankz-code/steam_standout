"""Validate dual-filter hypothesis: peak_week AND avg_daily together.

Outputs tables + matplotlib charts to verify that combining both velocity
dimensions dramatically improves screening precision before integrating
into the HTML report.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"
HISTORIES_DIR = DATA_DIR / "histories"
BACKDROP_FILE = DATA_DIR / "backdrop_games.parquet"
SUCCESS_FILE = CONFIG_DIR / "success_games.json"


# ── Reuse peak extraction logic from 10_generate_report ──────────────

def extract_peak_velocity_all(success_ids, backdrop_df=None):
    """Compute peak 7-day follower velocity from ALL history JSON files.

    Returns DataFrame with steamId, name, group, peak_week, avg_daily,
    peak_dtl, copies, revenue for every game with sufficient pre-launch history.
    """
    games = []
    history_files = sorted(HISTORIES_DIR.glob("*.json"))
    skipped = 0

    # Build name+revenue lookup from backdrop
    bd_lookup = {}
    if backdrop_df is not None:
        for _, r in backdrop_df.iterrows():
            bd_lookup[str(r["steamId"])] = {
                "name": r.get("name", ""),
                "copies": r.get("copiesSold", 0) or 0,
                "revenue": r.get("revenue", 0) or 0,
            }

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

        pre = df[df["days_to_launch"] > 0].copy()
        if len(pre) < 2:
            skipped += 1
            continue

        pre = pre.sort_values("date")
        pre["fol_diff"] = pre["followers"].diff()
        pre["date_diff"] = pre["date"].diff().dt.total_seconds() / 86400
        pre["daily_rate"] = pre["fol_diff"] / pre["date_diff"]

        # Onset artifact: null out first rate if game already had followers
        if pre.iloc[0]["followers"] > 100 and len(pre) > 1:
            pre.iloc[0, pre.columns.get_loc("daily_rate")] = np.nan

        rates = pre["daily_rate"].dropna()
        if len(rates) < 2:
            skipped += 1
            continue

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
        name = data.get("name", sid)
        copies = data.get("copiesSold", 0) or 0
        revenue = data.get("revenue", 0) or 0
        if not copies and sid in bd_lookup:
            copies = bd_lookup[sid].get("copies", 0)
            revenue = bd_lookup[sid].get("revenue", 0)

        games.append({
            "steamId": sid, "name": name, "group": group,
            "peak_week": round(peak, 1), "avg_daily": round(avg, 1),
            "peak_dtl": peak_dtl,
            "copies": copies, "revenue": revenue,
        })

    print(f"  Loaded {len(games)} games from {len(history_files)} files (skipped {skipped})")
    return pd.DataFrame(games)


def compute_cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = ((a.std() ** 2 + b.std() ** 2) / 2) ** 0.5
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


# ── Analysis Functions ────────────────────────────────────────────────

def analyze_cohens_d(gf):
    """Cohen's d for peak_week and avg_daily (success vs backdrop)."""
    s = gf[gf["group"] == "success"]
    b = gf[gf["group"] == "backdrop"]

    d_peak = compute_cohens_d(s["peak_week"], b["peak_week"])
    d_avg = compute_cohens_d(s["avg_daily"], b["avg_daily"])

    # Also on log-transformed values (more appropriate for skewed distributions)
    s_log_peak = np.log10(s["peak_week"].clip(lower=1))
    b_log_peak = np.log10(b["peak_week"].clip(lower=1))
    s_log_avg = np.log10(s["avg_daily"].clip(lower=1))
    b_log_avg = np.log10(b["avg_daily"].clip(lower=1))

    d_peak_log = compute_cohens_d(s_log_peak, b_log_peak)
    d_avg_log = compute_cohens_d(s_log_avg, b_log_avg)

    print("\n" + "=" * 70)
    print("COHEN'S d — SUCCESS vs BACKDROP")
    print("=" * 70)
    print(f"{'Metric':<25} {'d (raw)':>10} {'d (log10)':>10} {'S median':>12} {'B median':>12} {'Ratio':>8}")
    print("-" * 80)
    for label, d_raw, d_log, col in [
        ("Peak Week Velocity", d_peak, d_peak_log, "peak_week"),
        ("Avg Daily Velocity", d_avg, d_avg_log, "avg_daily"),
    ]:
        s_med = s[col].median()
        b_med = b[col].median()
        ratio = s_med / b_med if b_med != 0 else float("inf")
        print(f"{label:<25} {d_raw:>10.2f} {d_log:>10.2f} {s_med:>12.1f} {b_med:>12.1f} {ratio:>7.1f}x")

    return {"peak_d": d_peak, "avg_d": d_avg, "peak_d_log": d_peak_log, "avg_d_log": d_avg_log}


def analyze_spike_ratios(gf):
    """Spike ratio = peak / avg. Shows FPs are dominated by high-ratio games."""
    gf = gf.copy()
    gf["spike_ratio"] = gf["peak_week"] / gf["avg_daily"].clip(lower=0.1)

    s = gf[gf["group"] == "success"]
    b = gf[gf["group"] == "backdrop"]

    print("\n" + "=" * 70)
    print("SPIKE RATIO ANALYSIS (peak_week / avg_daily)")
    print("=" * 70)
    for label, subset in [("Success", s), ("Backdrop", b)]:
        print(f"\n  {label} (n={len(subset)}):")
        print(f"    Median ratio: {subset['spike_ratio'].median():.1f}x")
        print(f"    Mean ratio:   {subset['spike_ratio'].mean():.1f}x")
        print(f"    % with ratio > 50:  {(subset['spike_ratio'] > 50).mean():.0%}")
        print(f"    % with ratio > 100: {(subset['spike_ratio'] > 100).mean():.0%}")

    # Among FPs at peak >= 845
    high_peak_b = b[b["peak_week"] >= 845]
    if len(high_peak_b) > 0:
        print(f"\n  Backdrop games with peak >= 845 (n={len(high_peak_b)}):")
        print(f"    % with avg_daily <= 30:  {(high_peak_b['avg_daily'] <= 30).mean():.0%}")
        print(f"    % with avg_daily <= 50:  {(high_peak_b['avg_daily'] <= 50).mean():.0%}")
        print(f"    % with spike_ratio > 50: {(high_peak_b['spike_ratio'] > 50).mean():.0%}")


def analyze_dual_filter_grid(gf):
    """Dual-filter grid: precision/recall at (peak, avg) threshold pairs."""
    n_success = (gf["group"] == "success").sum()

    peak_thresholds = [200, 448, 600, 845, 1200, 1626, 3000, 4140]
    avg_thresholds = [0, 10, 20, 30, 50, 75, 100, 200]

    print("\n" + "=" * 70)
    print(f"DUAL-FILTER GRID (n_success={n_success}, total={len(gf)})")
    print("=" * 70)

    for pt in peak_thresholds:
        print(f"\n  --- Peak >= {pt} ---")
        print(f"  {'Avg >=':<8} {'Flagged':>8} {'Hits':>6} {'Hit Rate':>10} {'Catch':>8}")
        print(f"  {'-'*45}")
        for at in avg_thresholds:
            flagged = gf[(gf["peak_week"] >= pt) & (gf["avg_daily"] >= at)]
            tp = (flagged["group"] == "success").sum()
            n_f = len(flagged)
            prec = tp / n_f if n_f > 0 else 0
            rec = tp / n_success if n_success > 0 else 0
            marker = " <<<" if 0.20 <= prec <= 0.40 and rec >= 0.50 else ""
            print(f"  {at:>5}   {n_f:>8} {tp:>6} {prec:>9.1%} {rec:>7.1%}{marker}")

    return peak_thresholds, avg_thresholds


def design_new_tiers(gf):
    """Design 4 new strategy tiers using dual thresholds."""
    n_success = (gf["group"] == "success").sum()

    # Tier design: all tiers use a common avg floor, with escalating peak thresholds
    # We test a few avg floors to find the best common value
    print("\n" + "=" * 70)
    print("NEW 4-TIER STRATEGY DESIGN (dual thresholds)")
    print("=" * 70)

    # Find optimal avg floor: maximize hit rate improvement at sweet spot (peak>=845)
    # while keeping catch rate >= 55%
    print("\n  Testing avg floors at peak >= 845:")
    for avg_floor in [20, 30, 40, 50, 75]:
        flagged = gf[(gf["peak_week"] >= 845) & (gf["avg_daily"] >= avg_floor)]
        tp = (flagged["group"] == "success").sum()
        n_f = len(flagged)
        prec = tp / n_f if n_f > 0 else 0
        rec = tp / n_success if n_success > 0 else 0
        print(f"    avg >= {avg_floor:>3}: {n_f:>4} flagged, {prec:.1%} hit rate, {rec:.1%} catch")

    # Proposed tiers with avg >= 30 as the "spam filter"
    avg_floor = 30
    tiers = [
        ("撒网策略 Wide Net", 448, avg_floor),
        ("甜点策略 Sweet Spot", 845, avg_floor),
        ("精选策略 Tighter", 1626, 50),
        ("高确信策略 High Conviction", 4140, 100),
    ]

    print(f"\n  === PROPOSED TIERS (avg floor = {avg_floor} base) ===")
    print(f"  {'Tier':<30} {'Peak>=':<8} {'Avg>=':<7} {'Watch':>6} {'Hits':>5} {'Hit%':>7} {'Catch':>7}")
    print(f"  {'-'*75}")

    tier_results = []
    for label, peak_t, avg_t in tiers:
        flagged = gf[(gf["peak_week"] >= peak_t) & (gf["avg_daily"] >= avg_t)]
        tp = (flagged["group"] == "success").sum()
        n_f = len(flagged)
        prec = tp / n_f if n_f > 0 else 0
        rec = tp / n_success if n_success > 0 else 0
        fp = n_f - tp
        print(f"  {label:<30} {peak_t:>5}   {avg_t:>5}  {n_f:>6} {tp:>5} {prec:>6.1%} {rec:>6.1%}")
        tier_results.append({
            "label": label, "peak_t": peak_t, "avg_t": avg_t,
            "flagged": n_f, "hits": tp, "fp": fp,
            "precision": prec, "recall": rec,
        })

    # Compare with old single-metric tiers
    print(f"\n  === OLD TIERS (peak only, for comparison) ===")
    print(f"  {'Tier':<30} {'Peak>=':<8} {'Watch':>6} {'Hits':>5} {'Hit%':>7} {'Catch':>7}")
    print(f"  {'-'*65}")

    old_tiers = [("撒网", 448), ("甜点", 845), ("精选", 1626), ("高确信", 4140)]
    for label, peak_t in old_tiers:
        flagged = gf[gf["peak_week"] >= peak_t]
        tp = (flagged["group"] == "success").sum()
        n_f = len(flagged)
        prec = tp / n_f if n_f > 0 else 0
        rec = tp / n_success if n_success > 0 else 0
        print(f"  {label:<30} {peak_t:>5}   {n_f:>6} {tp:>5} {prec:>6.1%} {rec:>6.1%}")

    return tier_results


def analyze_false_positives(gf, tier_label="Sweet Spot", peak_t=845, avg_t=30):
    """Examine false positives at a given tier."""
    flagged = gf[(gf["peak_week"] >= peak_t) & (gf["avg_daily"] >= avg_t)]
    fps = flagged[flagged["group"] != "success"].sort_values("peak_week", ascending=False)
    hits = flagged[flagged["group"] == "success"].sort_values("peak_week", ascending=False)
    missed = gf[(gf["group"] == "success") & ~((gf["peak_week"] >= peak_t) & (gf["avg_daily"] >= avg_t))]

    print(f"\n{'=' * 70}")
    print(f"FALSE POSITIVE ANALYSIS — {tier_label} (peak>={peak_t}, avg>={avg_t})")
    print(f"{'=' * 70}")

    print(f"\n  HITS ({len(hits)} success games flagged):")
    for _, r in hits.iterrows():
        rev_str = f"${r['revenue']/1e6:.0f}M" if r['revenue'] > 0 else "N/A"
        print(f"    {r['name'][:40]:<42} peak={r['peak_week']:>8,.0f}  avg={r['avg_daily']:>6,.0f}  rev={rev_str}")

    print(f"\n  MISSED ({len(missed)} success games NOT flagged):")
    for _, r in missed.sort_values("peak_week", ascending=False).iterrows():
        reason = f"peak={r['peak_week']:.0f}" if r['peak_week'] < peak_t else f"avg={r['avg_daily']:.0f}"
        print(f"    {r['name'][:40]:<42} peak={r['peak_week']:>8,.0f}  avg={r['avg_daily']:>6,.0f}  ({reason})")

    print(f"\n  FALSE POSITIVES — Revenue distribution ({len(fps)} games):")
    if len(fps) > 0:
        print(f"    Median revenue: ${fps['revenue'].median()/1e6:.1f}M")
        print(f"    Revenue > $30M: {(fps['revenue'] > 30e6).sum()}/{len(fps)} ({(fps['revenue'] > 30e6).mean():.0%})")
        print(f"    Revenue > $50M: {(fps['revenue'] > 50e6).sum()}/{len(fps)} ({(fps['revenue'] > 50e6).mean():.0%})")
        print(f"    Copies > 1M:   {(fps['copies'] > 1e6).sum()}/{len(fps)} ({(fps['copies'] > 1e6).mean():.0%})")

    print(f"\n  TOP 20 FALSE POSITIVES:")
    print(f"  {'Name':<42} {'Peak':>8} {'Avg':>8} {'Copies':>12} {'Revenue':>12}")
    print(f"  {'-'*85}")
    for _, r in fps.head(20).iterrows():
        copies_str = f"{r['copies']:,.0f}" if r['copies'] > 0 else "N/A"
        rev_str = f"${r['revenue']/1e6:.1f}M" if r['revenue'] > 0 else "N/A"
        print(f"  {r['name'][:41]:<42} {r['peak_week']:>8,.0f} {r['avg_daily']:>8,.0f} {copies_str:>12} {rev_str:>12}")


# ── Plotting ──────────────────────────────────────────────────────────

def plot_dual_scatter(gf, peak_t=845, avg_t=30, filename="dual_scatter.png"):
    """Scatter plot: peak vs avg, colored by group, with threshold lines."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    s = gf[gf["group"] == "success"]
    b = gf[gf["group"] == "backdrop"]

    ax.scatter(b["avg_daily"].clip(lower=0.5), b["peak_week"].clip(lower=1),
               alpha=0.3, s=20, c="#94a3b8", label=f"Backdrop (n={len(b)})", zorder=2)
    ax.scatter(s["avg_daily"].clip(lower=0.5), s["peak_week"].clip(lower=1),
               alpha=0.8, s=60, c="#10b981", label=f"Success (n={len(s)})", zorder=3)

    # Label success games
    for _, r in s.iterrows():
        ax.annotate(r["name"][:20], (max(r["avg_daily"], 0.5), max(r["peak_week"], 1)),
                     fontsize=6, alpha=0.7, ha="left", va="bottom")

    # Threshold lines
    ax.axhline(y=peak_t, color="#ef4444", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Peak threshold ({peak_t})")
    ax.axvline(x=avg_t, color="#3b82f6", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Avg threshold ({avg_t})")

    # Shade the "target zone"
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([avg_t, xlim[1] * 2], peak_t, ylim[1] * 2,
                     alpha=0.05, color="#10b981", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avg Weekly Velocity (followers/day, 7d rolling median)")
    ax.set_ylabel("Peak Week Velocity (followers/day, 7d rolling max)")
    ax.set_title(f"Dual Filter: Peak >= {peak_t} AND Avg >= {avg_t}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


def plot_pr_comparison(gf, filename="dual_pr_curves.png"):
    """PR curves: single-metric vs dual-filter at different avg floors."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    n_success = (gf["group"] == "success").sum()

    # Generate thresholds
    peak_values = sorted(gf["peak_week"].unique())
    thresholds = sorted(set(
        list(np.percentile(gf["peak_week"].values, np.arange(5, 100, 2))) +
        [200, 300, 448, 600, 845, 1000, 1626, 2000, 3000, 4140, 5000]
    ))

    for avg_floor, color, label, ls in [
        (0, "#94a3b8", "Peak only (baseline)", "-"),
        (20, "#60a5fa", "Peak + avg >= 20", "--"),
        (30, "#3b82f6", "Peak + avg >= 30", "--"),
        (50, "#10b981", "Peak + avg >= 50", "-"),
        (100, "#ef4444", "Peak + avg >= 100", "-."),
    ]:
        precs, recs = [], []
        for t in thresholds:
            flagged = gf[(gf["peak_week"] >= t) & (gf["avg_daily"] >= avg_floor)]
            n_f = len(flagged)
            if n_f == 0:
                continue
            tp = (flagged["group"] == "success").sum()
            precs.append(tp / n_f)
            recs.append(tp / n_success)
        ax.plot(recs, precs, label=label, color=color, linestyle=ls, linewidth=2, alpha=0.8)

    ax.set_xlabel("Catch Rate (Recall)")
    ax.set_ylabel("Hit Rate (Precision)")
    ax.set_title("Precision-Recall: Single vs Dual Filter")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)

    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_spike_ratio(gf, filename="spike_ratio_hist.png"):
    """Histogram of spike_ratio for success vs backdrop."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    gf = gf.copy()
    gf["spike_ratio"] = gf["peak_week"] / gf["avg_daily"].clip(lower=0.1)
    gf["log_ratio"] = np.log10(gf["spike_ratio"].clip(lower=0.1))

    s = gf[gf["group"] == "success"]
    b = gf[gf["group"] == "backdrop"]

    bins = np.linspace(-0.5, 4, 50)
    ax.hist(b["log_ratio"], bins=bins, alpha=0.5, color="#94a3b8", label=f"Backdrop (n={len(b)})", density=True)
    ax.hist(s["log_ratio"], bins=bins, alpha=0.7, color="#10b981", label=f"Success (n={len(s)})", density=True)

    ax.axvline(np.log10(50), color="#ef4444", linestyle="--", alpha=0.7, label="Ratio = 50x")
    ax.set_xlabel("log10(Peak / Avg) — Spike Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Spike Ratio Distribution: Success vs Backdrop")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    with open(SUCCESS_FILE) as f:
        success_list = json.load(f)
    success_ids = {g["steamId"] for g in success_list}

    bd = pd.read_parquet(BACKDROP_FILE)
    bd["steamId"] = bd["steamId"].astype(str)

    print("Computing peak velocities from history files...")
    gf = extract_peak_velocity_all(success_ids, bd)

    print(f"\nDataset: {len(gf)} games total")
    print(f"  Success: {(gf['group'] == 'success').sum()}")
    print(f"  Backdrop: {(gf['group'] == 'backdrop').sum()}")

    # ── Analysis ──
    d_results = analyze_cohens_d(gf)
    analyze_spike_ratios(gf)
    analyze_dual_filter_grid(gf)
    tier_results = design_new_tiers(gf)
    analyze_false_positives(gf, "Sweet Spot", 845, 30)

    # ── Plots ──
    print("\nGenerating plots...")
    plot_dual_scatter(gf, peak_t=845, avg_t=30)
    plot_pr_comparison(gf)
    plot_spike_ratio(gf)

    print("\nDone! Check output/ for plots.")


if __name__ == "__main__":
    main()
