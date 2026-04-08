# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis project that identifies what makes certain Steam games "stand out" — games that massively exceeded commercial expectations. It uses the **Gamalytic API** to collect Steam game data, builds matched success/control groups, and runs statistical signal analysis on pre-launch features to find predictive patterns. The final output is a self-contained interactive HTML report hosted at GitHub Pages.

**Live report**: https://frankz-code.github.io/steam_standout/

## Running the Pipeline

Scripts run sequentially, each depending on the prior step's output. Run from the project root:

```bash
python src/01_collect_backdrop.py      # Collect all Steam games (2023–2026) → data/backdrop_games.parquet
python src/02_match_control_group.py   # Interactive: match controls to success games → config/control_group.json
python src/03_collect_histories.py     # Fetch history for success+control → data/histories/*.json (with include_pre_release_history=True)
python src/03b_collect_backdrop_sample.py  # Stratified sample of backdrop histories
python src/03c_refetch_backdrop_prerelease.py  # Re-fetch backdrop with include_pre_release_history=True (CRITICAL — see note)
python src/04_build_analysis_set.py    # Compute pre-launch features → data/analysis_features.parquet
python src/06_forward_analysis.py      # Forward observations → data/forward_observations.parquet
python src/10_generate_report.py       # Generate HTML report → output/report.html + output/report_text.md
```

Step 02 is **interactive**. Steps 01, 03, 03b, 03c make API calls and support resume.

**Important**: `03b` originally omitted `include_pre_release_history=True`, causing most backdrop games to lack pre-launch data. `03c` fixes this by re-fetching with the flag. Always use `include_pre_release_history=True` when fetching game details for velocity analysis.

## Dependencies

```bash
pip install -r requirements.txt   # requests, pandas, pyarrow, matplotlib, seaborn, python-dotenv
```

Requires a `.env` file with `GAMALYTIC_API_KEY`.

## Architecture

### Data Flow
```
Gamalytic API → backdrop_games.parquet (61K games)
             → histories/*.json (4,872 files, 3,819 with pre-launch data)
             → analysis_features.parquet (330 success+control)
             → forward_observations.parquet (330 success+control timeseries)
             → 10_generate_report.py reads ALL of above
             → output/report.html (~1MB self-contained)
```

### Key Components

- **`src/api_client.py`** — `GamalyticClient` wrapper with per-endpoint rate limiting (600/240/30 req/min tiers), exponential backoff retry. Supports `include_pre_release_history` parameter on `get_game_details()`.
- **`config/success_games.json`** — Hand-curated list of 30 "standout" games with Steam IDs.
- **`config/control_group.json`** — Maps each success game to ~10 matched control game IDs (Jaccard tag similarity 60%, date 20%, price 20%).
- **`data/histories/`** — One JSON file per game (named by Steam ID). History entries use `timeStamp` key (not `date`). Contains `history` array with followers, wishlists, revenue timeseries.

### Report Generation (src/10_generate_report.py, ~2100 lines)

The main report script. It has two data paths:
- **`peak_data`**: From `forward_observations.parquet` — 330 success+control games. Used for Section 1 charts (trajectories, Cohen's d on matched controls).
- **`peak_data_full`**: From `extract_peak_week_from_histories()` — reads ALL 4,872 history JSONs, computes peak/avg velocity for ~3,810 games with pre-launch data. Used for Section 2 strategy evaluation, scatter charts, revenue analysis.

Key extraction functions:
- `extract_peak_week_from_histories()` → peak_week (max of 7d rolling mean), avg_daily (MEDIAN of 7d rolling mean, despite the name), revenue
- `extract_strategy_tiers()` → 3-tier dual-threshold strategy (peak AND avg)
- `extract_strategy_quality()` → revenue-based evaluation ($10M rate, $50M rate, Top-100, Lift)
- `extract_threshold_sweep(data, avg_floor)` → PR curve with avg pre-filter
- `extract_scatter_data(af, names, peak_data_full)` → followers-vs-revenue from full backdrop

### Dual-Metric Screening (v4)

Strategy evaluation uses **two velocity dimensions**:
- **Peak Week Velocity**: max of 7-day rolling mean of daily follower growth (pre-launch)
- **Median Weekly Velocity** (field: `avg_daily`): median of 7-day rolling mean — captures sustained interest vs one-off spikes

Three tiers, calibrated via stepped percentiles on the 3,783-game backdrop:

| Tier | Peak ≥ | Avg ≥ | Watch | Hit Rate | Catch | Lift |
|------|--------|-------|-------|----------|-------|------|
| 广角 | 400 | 20 | ~335 | 6% | 78% | 9x |
| 精选 | 900 | 40 | ~135 | 13% | 63% | 18x |
| 高确信 | 1900 | 70 | ~58 | 19% | 41% | 27x |

### Data Quality Notes

- **Follower counts = true Steam API data**; wishlist counts and revenue = Gamalytic estimates.
- **`avg_daily` is a median**, not a mean. It's the median of 7-day rolling mean values across all pre-launch observation points. This correctly identifies "spike-only" games (median ≈ 0).
- **Onset artifact handling**: if first history entry has >100 followers, the first growth rate is nulled to prevent fake spikes from mid-tracking starts.
- **3 success games lack pre-launch data** (<7 days): PEAK (shadow drop), Dark and Darker, Content Warning.

## Conventions

- Steam IDs are always stored and compared as **strings**.
- Timestamps from the API are **epoch milliseconds**; converted to pandas datetime.
- All file paths use `pathlib.Path` with `PROJECT_ROOT = Path(__file__).parent.parent`.
- Parquet is the intermediate data format (not CSV).
- Chinese game name mapping via `CN` dict and `cn()` function in `10_generate_report.py`.
- Report terminology: 命中率 = Hit Rate, 不漏球率 = Catch Rate, 看走眼？ = False Positive.

## Documentation

- **`report3.md`** — Current v4 framework and implementation details (chart index, function map, tier design)
- **`report2.md`** — Historical v2/v3 framework + v3→v4 changelog
- **`output/report_text_v4.md`** — Extracted report text (user-polished Chinese copy)
