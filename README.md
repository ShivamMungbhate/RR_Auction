# SupeRR Selector S2 - Pressure-rating dashboard

Streamlit app that **only reads precomputed files** from `results_pool/`. It never re-runs a model and makes no network calls.

## Folder layout (commit exactly this)
```
superr_dashboard/
├── app.py
├── requirements.txt
├── README.md
└── results_pool/
    ├── bat_ratings.csv            bowl_ratings.csv
    ├── bat_bootstrap.csv          bowl_bootstrap.csv
    ├── bat_rank_stability.csv     bowl_rank_stability.csv
    ├── pressure_validation_batting.csv   pressure_validation_bowling.csv
    ├── chart_pressure_by_over.png
    ├── chart_bat_pressure_vs_normal.png  chart_bowl_pressure_vs_normal.png
    ├── meta.json      (optional - see below)
    └── rr_fit.csv     (optional - fills the "RR fit" page)
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repo and push the folder above (repo root = the folder containing `app.py`).
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**.
3. Pick the repo and branch, set **Main file path** to `app.py`, click **Deploy**.
4. Every `git push` redeploys automatically.

## Updating on 17 Nov (no code change)
Replace the files inside `results_pool/` with the new run (same file names), commit, push. The app reloads; cached data is keyed on file modification time. The `DATA_DIR` constant at the top of `app.py` is the only path to change if you rename the folder.

## Optional files
- `results_pool/meta.json` - override figures quoted from the write-up and set the footer date, e.g.
  `{"last_updated": "17 Nov 2026", "matches": 812, "split_half_bat": 0.14, "split_half_bowl": 0.08}`
  Allowed keys: `deliveries, matches, ipl_matches, t20i_major_matches, t20i_assoc_matches, wp_logloss, coin_flip_logloss, wicket_cost, bootstrap_resamples, bootstrap_seed, shrinkage_k, split_half_bat, split_half_bowl, score_within_2, score_within_5, last_updated`.
- `results_pool/rr_fit.csv` - columns `RR need, Candidate(s), Why they fit, Notes`.

## Expected but not found in the result files
These appear in the app from `FACTS` in `app.py` (taken from the project brief) or as placeholders:
- **Split-half correlations** (0.14 / 0.08): no file contains them.
- **Data counts, WP log-loss, wicket costs, score-check percentages**: not in any CSV.
- **Pool-filter details** (2026 squads, retired, Pakistan/Bangladesh): the files contain only the final pool; names removed are not listed, so the app shows the rules as text.
- **Nationality / overseas flag / playing role**: absent, so the RR-fit page is placeholders and cannot filter overseas players.
- **Last-updated timestamp**: no field; the app uses file modification times (on Streamlit Cloud this is the deploy time, so set `last_updated` in `meta.json`).
- `latest_ipl_team` is blank for 12 batters and 10 bowlers.
- Rank-stability files list only the main top 10; blanks mean not ranked in that variant.