
import re
import argparse
import glob
import os
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

FULL_MEMBERS = {
    "India", "Australia", "England", "South Africa", "New Zealand",
    "Pakistan", "Sri Lanka", "Bangladesh", "West Indies",
    "Afghanistan", "Ireland", "Zimbabwe",
}

OUTCOMES = ["0", "1", "2", "3", "4", "5", "6", "W"]
OUT_RUNS = {
    "0": 0, "1": 1, "2": 2, "3": 3,
    "4": 4, "5": 5, "6": 6, "W": 0,
}

DEFAULT_K = 120

DEFAULT_DATA_DIR = "."
DEFAULT_OUT_DIR = "results_a"
DEFAULT_BOOTSTRAP = 500
DEFAULT_EXCLUDE_IDS = "142,167,143,278,305,1318,2485,2490,2671,2681,2878,2921,3024,3056,3489,4658,4773,4926,5837,5979,6355,6651,7344,7364,7690,7699,7900,7901,7903,7905,8049,8380,4373,1596"
DEFAULT_UNAVAILABLE_FILE = "ipl2027_unavailable.csv"


def find(data_dir, name):
    hits = (
        glob.glob(os.path.join(data_dir, name + "*.csv"))
        or glob.glob(
            os.path.join(data_dir, "**", name + "*.csv"),
            recursive=True,
        )
    )
    if not hits:
        raise FileNotFoundError(
            f"Could not find {name}*.csv inside {data_dir}"
        )
    return sorted(hits)[0]


def load(data_dir):
    bb = pd.read_csv(
        find(data_dir, "ball_by_ball"), low_memory=False
    )
    matches = pd.read_csv(
        find(data_dir, "matches"), low_memory=False
    )
    competitions = pd.read_csv(
        find(data_dir, "competitions"), low_memory=False
    )
    mapping = pd.read_csv(
        find(data_dir, "competition_team_player_mapping"),
        low_memory=False,
    )
    return bb, matches, competitions, mapping


def diagnostics(bb):
    print("\n========== DATA DIAGNOSTICS ==========")
    print("Shape:", bb.shape)
    print("\nInnings:\n", bb["innings"].value_counts().to_string())
    print(
        "\nExtras:\n",
        bb["extras_type"].value_counts(dropna=False).to_string(),
    )
    print(
        "\nDismissals:\n",
        bb["dismissal_type"].value_counts(dropna=False).to_string(),
    )
    print("======================================\n")


def as_bool(series):
    """Safely convert booleans and common CSV boolean strings."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def prepare(bb, matches, competitions):
    bb = bb[bb["innings"].isin([1, 2])].copy()

    match_cols = [
        "match_id", "comp_id", "match_date", "home_team",
        "away_team", "win_team_id", "knock_out",
        "first_inn_revised_overs", "second_inn_revised_overs",
        "revised_target", "result",
    ]
    match_cols = [c for c in match_cols if c in matches.columns]

    comp_cols = [
        "comp_id", "comp_name", "match_format", "season"
    ]
    comp_cols = [c for c in comp_cols if c in competitions.columns]

    mm = matches[match_cols].merge(
        competitions[comp_cols],
        on="comp_id",
        how="left",
    )

    mm["match_date"] = pd.to_datetime(
        mm["match_date"], errors="coerce"
    )

    def competition_tier(row):
        comp_name = str(row.get("comp_name", ""))
        fmt = str(row.get("match_format", "")).upper().strip()

        if "IPL" in comp_name.upper():
            return "IPL"

        if fmt == "T20I":
            if (
                row.get("home_team") in FULL_MEMBERS
                and row.get("away_team") in FULL_MEMBERS
            ):
                return "T20I_major"
            return "T20I_assoc"

        return "OTHER"

    mm["ctx"] = mm.apply(competition_tier, axis=1)

    bb = bb.merge(mm, on="match_id", how="inner")
    bb = bb[bb["win_team_id"].notna()].copy()

    et = (
        bb["extras_type"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    bb["is_wide"] = et.eq("wide")

    bb["is_nb"] = et.isin([
        "overstep", "noball", "no ball", "no-ball"
    ])

    bb["is_bye"] = et.isin([
        "byes", "legbyes", "bye", "legbye",
        "leg-bye", "leg-byes"
    ])

    known = {
        "", "wide", "overstep", "noball", "no ball",
        "no-ball", "byes", "legbyes", "bye",
        "legbye", "leg-bye", "leg-byes",
        "other", "penalty",
    }

    unknown = sorted(set(et.unique()) - known)

    if unknown:
        print("\nWARNING: Unknown extras types:", unknown)

    extra_map = {
        "wide": 1,
        "overstep": 1,
        "noball": 1,
        "no ball": 1,
        "no-ball": 1,
        "byes": 1,
        "legbyes": 1,
        "bye": 1,
        "legbye": 1,
        "leg-bye": 1,
        "leg-byes": 1,
        "penalty": 5,
    }

    bb["assumed_extras"] = et.map(extra_map).fillna(0)

    runs = pd.to_numeric(
        bb["runs"], errors="coerce"
    ).fillna(0)

    bb["runs"] = runs

    bb["legal"] = ~(
        bb["is_wide"] | bb["is_nb"]
    )

    if "extras_2_runs" in bb.columns:
        ex2 = pd.to_numeric(bb["extras_2_runs"], errors="coerce").fillna(0).values
    else:
        ex2 = np.zeros(len(bb))
    ex2_bat = np.where(bb["is_nb"].values | et.eq("other").values, ex2, 0)

    bb["team_runs"] = (
        bb["runs"] + bb["assumed_extras"] + ex2
    )

    bb["bat_runs"] = np.where(
        bb["is_wide"] | bb["is_bye"],
        0,
        bb["runs"] + ex2_bat,
    )

    bb["bowl_runs"] = (
        bb["bat_runs"]
        + np.where(
            bb["is_wide"] | bb["is_nb"],
            bb["assumed_extras"],
            0,
        )
    )

    print("\n========== EXTRAS AUDIT ==========")

    print("\nExtras type counts:")
    print(
        bb["extras_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nAssumed extras by type:")
    print(
        bb.groupby(
            "extras_type",
            dropna=False
        )["assumed_extras"]
        .agg(["count", "sum", "mean"])
        .to_string()
    )

    print("\nRuns field by extras type:")
    print(
        bb.groupby(
            "extras_type",
            dropna=False
        )["runs"]
        .agg(
            count="count",
            min="min",
            median="median",
            max="max",
            total="sum",
        )
        .to_string()
    )
    if "extras_2_runs" in bb.columns:
        extra2 = pd.to_numeric(
            bb["extras_2_runs"],
            errors="coerce"
        ).fillna(0)

        print("\nextras_2_runs value counts:")
        print(
            extra2.value_counts()
            .sort_index()
            .to_string()
        )

        nonzero = bb.loc[extra2.ne(0)].copy()

        print("\nNonzero extras_2_runs rows:", len(nonzero))

        if len(nonzero):
            cols = [
                c for c in [
                    "match_id", "innings", "over_ball_num",
                    "runs", "extras_type", "extras1_runs",
                    "extras_2_runs", "result",
                ]
                if c in nonzero.columns
            ]

            print(
                nonzero[cols]
                .sort_values(["match_id", "innings"])
                .to_string(index=False)
            )
    else:
        print("\nextras_2_runs column not found.")

    print("\n==================================")

    dt = (
        bb["dismissal_type"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    is_wicket = as_bool(bb["is_wicket"])

    retired_hurt = dt.str.contains(
        "retired hurt", regex=False
    )

    bb["wk"] = is_wicket & ~retired_hurt

  
    non_bowler_dismissal = dt.str.contains(
        r"run\s*out|obstr(?:uct|act)\w*|timed\s*out|retir(?:e|ed)\s*out",
        regex=True,
    )

    bb["bowler_wk"] = (
        bb["wk"] & ~non_bowler_dismissal
    )

    dismissal_audit = (
        bb.loc[bb["wk"]]
        .groupby("dismissal_type", dropna=False)
        .agg(
            dismissals=("wk", "size"),
            bowler_credited=("bowler_wk", "sum"),
        )
    )

    print("\nDISMISSAL ATTRIBUTION AUDIT:")
    print(dismissal_audit.to_string())

    sort_cols = [
        c for c in [
            "match_id",
            "innings",
            "innings_ball_num",
            "over_ball_num",
        ]
        if c in bb.columns
    ]

    bb = bb.sort_values(
        sort_cols,
        kind="stable",
    ).reset_index(drop=True)

    group = bb.groupby(
        ["match_id", "innings"],
        sort=False,
    )

    bb["runs_before"] = (
        group["team_runs"].cumsum()
        - bb["team_runs"]
    )

    bb["wk_before"] = (
        group["wk"].cumsum()
        - bb["wk"].astype(int)
    )

    bb["balls_before"] = (
        group["legal"].cumsum()
        - bb["legal"].astype(int)
    )

    bb["over_idx"] = (
        (bb["balls_before"] // 6)
        .clip(0, 19)
        .astype(int)
    )

    first_total = (
        bb[bb["innings"] == 1]
        .groupby("match_id")["team_runs"]
        .sum()
        .rename("inn1_total")
    )

    bb = bb.merge(
        first_total,
        on="match_id",
        how="left",
    )

    revised_target = pd.to_numeric(
        bb["revised_target"],
        errors="coerce",
    )

    bb["target"] = np.where(
        revised_target.notna(),
        revised_target,
        bb["inn1_total"] + 1,
    )

    first_overs = pd.to_numeric(
        bb["first_inn_revised_overs"],
        errors="coerce",
    ).fillna(20)

    second_overs = pd.to_numeric(
        bb["second_inn_revised_overs"],
        errors="coerce",
    ).fillna(20)

    bb["max_balls"] = np.where(
        bb["innings"] == 1,
        first_overs * 6,
        second_overs * 6,
    )

    bb["won"] = (
        bb["batting_team_id"] == bb["win_team_id"]
    ).astype(int)

    bb["needed"] = np.where(
        bb["innings"] == 2,
        bb["target"] - bb["runs_before"],
        np.nan,
    )

    bb["balls_left"] = (
        bb["max_balls"] - bb["balls_before"]
    ).clip(lower=0)

    bb["rrr"] = np.where(
        bb["innings"] == 2,
        bb["needed"] * 6
        / bb["balls_left"].clip(lower=1),
        np.nan,
    )
    bb["ctx_code"] = (
        bb["ctx"]
        .map({
            "IPL": 0,
            "T20I_major": 1,
            "T20I_assoc": 2,
            "OTHER": 3,
        })
        .fillna(3)
    )

    bb["phase"] = pd.cut(
        bb["over_idx"],
        [-1, 5, 14, 19],
        labels=["PP", "MID", "DEATH"],
    ).astype(str)

    

    print("\n========== SCORE VALIDATION ==========")

    totals = (
        bb.groupby(
            ["match_id", "innings"]
        )["team_runs"]
        .sum()
        .unstack()
    )

    if 1 in totals.columns and 2 in totals.columns:

        score_data = totals[[1, 2]].copy()
        score_data.columns = [
            "inn1_total", "inn2_total"
        ]

        meta_cols = [
            c for c in [
                "match_id",
                "result",
                "revised_target",
                "first_inn_revised_overs",
                "second_inn_revised_overs",
                "ctx",
            ]
            if c in bb.columns
        ]

        meta = (
            bb[meta_cols]
            .drop_duplicates("match_id")
            .set_index("match_id")
        )

        score_data = score_data.join(meta, how="left")

        if "result" in score_data.columns:

            result_text = (
                score_data["result"]
                .fillna("")
                .astype(str)
            )

            
            is_dls = result_text.str.contains(
                r"\bDLS\b|\bD/L\b|\bDuckworth\b",
                case=False,
                regex=True,
            )

            run_margin = pd.to_numeric(
                result_text.str.extract(
                    r"won by\s+(\d+)\s+run",
                    flags=re.IGNORECASE,
                )[0],
                errors="coerce",
            )

            score_data["run_margin"] = run_margin
            score_data["is_dls"] = is_dls

           
            run_matches = score_data.loc[
                run_margin.notna() & ~is_dls
            ].copy()

            run_matches["score_difference"] = (
                run_matches["inn1_total"]
                - run_matches["inn2_total"]
            )

            run_matches["score_error"] = (
                run_matches["score_difference"]
                - run_matches["run_margin"]
            )

            if len(run_matches):
                errors = run_matches["score_error"]

                print("\nNON-DLS RUN-WIN SCORE CHECK:")
                print(f"Matches checked: {len(errors)}")
                print(f"Mean error: {errors.mean():.2f}")
                print(f"Median error: {errors.median():.2f}")
                print(f"Mean absolute error: {errors.abs().mean():.2f}")
                print(
                    f"Within ±1: "
                    f"{errors.abs().le(1).mean():.1%}"
                )
                print(
                    f"Within ±2: "
                    f"{errors.abs().le(2).mean():.1%}"
                )
                print(
                    f"Within ±5: "
                    f"{errors.abs().le(5).mean():.1%}"
                )

                print("\nLargest 15 non-DLS mismatches:")

                show_cols = [
                    c for c in [
                        "inn1_total",
                        "inn2_total",
                        "score_difference",
                        "run_margin",
                        "score_error",
                        "result",
                        "ctx",
                    ]
                    if c in run_matches.columns
                ]

                print(
                    run_matches.assign(
                        abs_error=run_matches["score_error"].abs()
                    )
                    .sort_values(
                        "abs_error",
                        ascending=False
                    )[show_cols]
                    .head(15)
                    .to_string()
                )
            else:
                print("No ordinary run-win matches available for validation.")

            print(
                "\nDLS run-win matches excluded from raw score check:",
                int((run_margin.notna() & is_dls).sum()),
            )

           
            wicket_margin = pd.to_numeric(
                result_text.str.extract(
                    r"won by\s+(\d+)\s+wicket",
                    flags=re.IGNORECASE,
                )[0],
                errors="coerce",
            )

            print(
                "Matches won by wickets:",
                int(wicket_margin.notna().sum()),
            )

           
            revised_target_num = pd.to_numeric(
                score_data.get(
                    "revised_target",
                    pd.Series(index=score_data.index, dtype=float),
                ),
                errors="coerce",
            )

            first_overs_num = pd.to_numeric(
                score_data.get(
                    "first_inn_revised_overs",
                    pd.Series(index=score_data.index, dtype=float),
                ),
                errors="coerce",
            )

            second_overs_num = pd.to_numeric(
                score_data.get(
                    "second_inn_revised_overs",
                    pd.Series(index=score_data.index, dtype=float),
                ),
                errors="coerce",
            )

            revised_mask = (
                revised_target_num.notna()
                | (
                    first_overs_num.notna()
                    & first_overs_num.lt(20)
                )
                | (
                    second_overs_num.notna()
                    & second_overs_num.lt(20)
                )
            )

            revised = score_data.loc[revised_mask]

            print(
                "Matches with revised target/reduced overs:",
                len(revised),
            )

            if len(revised):
                revised_cols = [
                    c for c in [
                        "inn1_total",
                        "inn2_total",
                        "result",
                        "revised_target",
                        "first_inn_revised_overs",
                        "second_inn_revised_overs",
                        "ctx",
                    ]
                    if c in revised.columns
                ]

                print(
                    revised[revised_cols]
                    .head(20)
                    .to_string()
                )

        else:
            print("No result column available for score validation.")

    else:
        print("Could not find both innings in reconstructed totals.")

    print("\n======================================")

    return bb


def make_classifier_1():
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=150,
        min_samples_leaf=300,
        l2_regularization=1.0,
        monotonic_cst=[1, 0, 0],
        random_state=0,
    )


class WP1:
    """First-innings win-probability model."""

    def fit(self, d):
        d = d.reset_index(drop=True)

        x = np.column_stack([
            d["balls_before"].values,
            d["wk_before"].values,
            d["ctx_code"].values,
        ])

        remaining = (
            d["inn1_total"].values - d["runs_before"].values
        )

        groups = d["match_id"].values
        unique_groups = pd.Series(groups).nunique()
        oof_remaining = np.full(len(d), np.nan)

        inner_splits = min(3, unique_groups)

        if inner_splits >= 2:
            gkf = GroupKFold(n_splits=inner_splits)

            for tr, va in gkf.split(x, remaining, groups):
                reg = make_regressor()
                reg.fit(x[tr], remaining[tr])
                oof_remaining[va] = reg.predict(x[va])

        if np.isnan(oof_remaining).any():
            fallback = np.nanmean(remaining)
            oof_remaining = np.where(
                np.isnan(oof_remaining),
                fallback,
                oof_remaining,
            )

        self.er = make_regressor().fit(x, remaining)

        projected_oof = (
            d["runs_before"].values + oof_remaining
        )

        x_classifier = np.column_stack([
            projected_oof,
            d["balls_before"].values,
            d["ctx_code"].values,
        ])

        self.clf = make_classifier_1().fit(
            x_classifier,
            d["won"].values,
        )
        return self

    def predict(self, runs, balls, wk, ctx):
        x_reg = np.column_stack([balls, wk, ctx])
        remaining = self.er.predict(x_reg)
        remaining = np.where(wk >= 10, 0, remaining)
        projected = runs + remaining

        x_clf = np.column_stack([projected, balls, ctx])
        return self.clf.predict_proba(x_clf)[:, 1]


class WP2:
    """Chase win-probability model with monotonic constraints."""

    @staticmethod
    def features(needed, left, wk, ctx):
        left = np.maximum(np.asarray(left), 1)
        needed = np.asarray(needed)
        wk = np.asarray(wk)
        ctx = np.asarray(ctx)

        return np.column_stack([
            needed,
            left,
            10 - wk,
            needed * 6 / left,
            ctx,
        ])

    def fit(self, d):
        needed = (
            d["target"].values - d["runs_before"].values
        )
        left = (
            d["max_balls"].values - d["balls_before"].values
        )

        x = self.features(
            needed,
            left,
            d["wk_before"].values,
            d["ctx_code"].values,
        )

        self.clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=300,
            l2_regularization=1.0,
            monotonic_cst=[-1, 1, 1, -1, 0],
            random_state=0,
        ).fit(x, d["won"].values)

        return self

    def predict(self, needed, left, wk, ctx):
        x = self.features(needed, left, wk, ctx)
        p = self.clf.predict_proba(x)[:, 1]

        p = np.where(needed <= 0, 1.0, p)
        p = np.where(
            (needed > 0) & ((wk >= 10) | (left <= 0)),
            0.0,
            p,
        )
        return p


def wp_at(model, innings, d, runs_added=0, wicket_added=0, after=False):
    """Win probability of the batting team. after=False: state BEFORE the ball; after=True: state after a hypothetical outcome."""
    runs = d["runs_before"].values + runs_added
    balls = d["balls_before"].values + (1 if after else 0)
    wickets = d["wk_before"].values + wicket_added
    ctx = d["ctx_code"].values

    if innings == 1:
        return model.predict(runs, balls, wickets, ctx)

    needed = d["target"].values - runs
    left = d["max_balls"].values - balls

    return model.predict(needed, left, wickets, ctx)


def make_regressor():
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=5.0,
        random_state=42,
    )


def win_prob_and_leverage(bb, folds=5):
    bb = bb.copy()

    bb["wp"] = np.nan
    for outcome in OUTCOMES:
        bb[f"wp_{outcome}"] = np.nan

    for innings in (1, 2):
        idx = np.where(bb["innings"].values == innings)[0]
        sub = bb.iloc[idx].reset_index(drop=True)

        groups = sub["match_id"].values
        n_groups = pd.Series(groups).nunique()
        n_splits = min(folds, n_groups)

        if n_splits < 2:
            raise ValueError(
                f"Not enough distinct matches for innings {innings}"
            )

        gkf = GroupKFold(n_splits=n_splits)

        for train_idx, test_idx in gkf.split(sub, groups=groups):
            train = sub.iloc[train_idx]
            test = sub.iloc[test_idx]

            model = (
                WP1() if innings == 1 else WP2()
            ).fit(train)

            global_test_idx = idx[test_idx]

            bb.loc[global_test_idx, "wp"] = wp_at(
                model, innings, test
            )

            for outcome in OUTCOMES:
                bb.loc[
                    global_test_idx, f"wp_{outcome}"
                ] = wp_at(
                    model,
                    innings,
                    test,
                    runs_added=OUT_RUNS[outcome],
                    wicket_added=(1 if outcome == "W" else 0),
                    after=True,
                )

    legal = bb[bb["legal"]].copy()

    categories = np.select(
        [
            legal["wk"],
            legal["team_runs"] == 0,
            legal["team_runs"] == 1,
            legal["team_runs"] == 2,
            legal["team_runs"] == 3,
            legal["team_runs"] == 4,
            legal["team_runs"] == 5,
            legal["team_runs"] >= 6,
        ],
        ["W", "0", "1", "2", "3", "4", "5", "6"],
        default="0",
    )

    legal["outcome"] = categories

    distribution = pd.crosstab(
        [legal["innings"], legal["over_idx"]],
        legal["outcome"],
        normalize="index",
    ).reindex(columns=OUTCOMES, fill_value=0)

    row_index = pd.MultiIndex.from_arrays([
        bb["innings"],
        bb["over_idx"],
    ])

    probabilities = (
        distribution.reindex(row_index)
        .fillna(0)
        .values
    )

    after = np.column_stack([
        bb[f"wp_{outcome}"].values
        for outcome in OUTCOMES
    ])

    swing = np.abs(after - bb["wp"].values[:, None])

    bb["leverage"] = np.sum(
        probabilities * swing,
        axis=1,
    )

    bb.loc[bb["innings"] == 1, "leverage"] = np.nan

    return bb


def first_innings_pressure(bb):
    """
    Role-specific first-innings pressure (percentile ranks, 0-1):
      Batters : 0.5 * wickets in last 12 legal balls (collapse) + 0.5 * runs below par at that over.
      Bowlers : 0.5 * runs conceded in last 12 legal balls (under attack) + 0.5 * runs above par at that over.
    Par = median first-innings score at that over within the competition context.
    Chase pressure (both roles) = expected absolute win-probability swing (leverage).
    """
    d = bb.copy()
    d["recent_wickets"] = 0.0
    d["recent_runs"] = 0.0

    for _, indexes in d.groupby(["match_id", "innings"], sort=False).groups.items():
        part = d.loc[indexes]
        legal_mask = part["legal"].fillna(False).to_numpy(dtype=bool)
        if not legal_mask.any():
            continue
        legal_wk = part.loc[legal_mask, "wk"].fillna(0).astype(float).to_numpy()
        legal_rn = part.loc[legal_mask, "team_runs"].fillna(0).astype(float).to_numpy()
        pw = np.concatenate(([0.0], np.cumsum(legal_wk)))
        pr = np.concatenate(([0.0], np.cumsum(legal_rn)))
        legal_before = np.cumsum(legal_mask) - legal_mask.astype(int)
        start = np.maximum(0, legal_before - 12)
        d.loc[part.index, "recent_wickets"] = pw[legal_before] - pw[start]
        d.loc[part.index, "recent_runs"] = pr[legal_before] - pr[start]

    first_mask = d["innings"] == 1
    par = (
        d.loc[first_mask].groupby(["ctx", "over_idx"])["runs_before"]
        .median().rename("par_runs")
    )
    d = d.join(par, on=["ctx", "over_idx"])
    d["par_runs"] = d["par_runs"].fillna(d["runs_before"])
    d["below_par"] = (d["par_runs"] - d["runs_before"]).clip(lower=0)
    d["above_par"] = (d["runs_before"] - d["par_runs"]).clip(lower=0)

    first = d.loc[first_mask]
    ranks = {
        "collapse_rank": first.groupby("ctx")["recent_wickets"].rank(pct=True),
        "below_par_rank": first.groupby(["ctx", "over_idx"])["below_par"].rank(pct=True),
        "attack_rank": first.groupby("ctx")["recent_runs"].rank(pct=True),
        "above_par_rank": first.groupby(["ctx", "over_idx"])["above_par"].rank(pct=True),
    }
    for name, r in ranks.items():
        d[name] = np.nan
        d.loc[first_mask, name] = r

    d["pressure_bat"] = np.nan
    d["pressure_bowl"] = np.nan
    d.loc[first_mask, "pressure_bat"] = 0.5 * d.loc[first_mask, "collapse_rank"] + 0.5 * d.loc[first_mask, "below_par_rank"]
    d.loc[first_mask, "pressure_bowl"] = 0.5 * d.loc[first_mask, "attack_rank"] + 0.5 * d.loc[first_mask, "above_par_rank"]

    chase_mask = d["innings"] == 2
    d.loc[chase_mask, "pressure_bat"] = d.loc[chase_mask, "leverage"]
    d.loc[chase_mask, "pressure_bowl"] = d.loc[chase_mask, "leverage"]
    return d


def assign_pressure_flags(bb, q=0.75):
    """Flag the top (1-q) of pressure within each innings, separately for batting and bowling."""
    bb = bb.copy()
    for role in ("bat", "bowl"):
        col, flag = f"pressure_{role}", f"high_{role}"
        bb[flag] = False
        for innings in (1, 2):
            mask = (bb["innings"] == innings) & bb[col].notna()
            if mask.sum() == 0:
                continue
            thr = bb.loc[mask, col].quantile(q)
            bb.loc[mask, flag] = bb.loc[mask, col] >= thr
    bb["high"] = bb["high_bat"]
    return bb


def wicket_cost_in_runs(bb):
    """Estimate wicket cost; flag values that hit the clipping boundary."""
    result = {}
    legal = bb[bb["legal"]].copy()

    for phase, g in legal.groupby("phase"):
        wicket_swing = (g["wp_0"] - g["wp_W"]).mean()
        run_swing = ((g["wp_6"] - g["wp_0"]) / 6).mean()

        raw = (
            wicket_swing / run_swing
            if pd.notna(run_swing) and run_swing > 0
            else np.nan
        )

        clipped = (
            float(np.clip(raw, 2, 25))
            if pd.notna(raw)
            else 10.0
        )

        result[phase] = clipped

        if pd.notna(raw):
            print(
                f"Wicket cost {phase}: raw={raw:.2f}, "
                f"used={clipped:.2f}"
            )
            if raw < 2 or raw > 25:
                print(
                    "  WARNING: estimate reached a clipping boundary."
                )
        else:
            print(
                f"Wicket cost {phase}: unavailable; used=10.00"
            )

    return result


def smoothed_group_mean(frame, key, value, min_count=30):
    """
    Hierarchical baseline:
      1. Detailed context, innings, over, wickets and bowling type.
      2. Context, innings, phase and wickets.
      3. Context, innings and phase.
      4. Context and innings.
      5. Overall mean.
    """

    out = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    levels = [
        key,
        ["ctx", "innings", "phase", "wk_bucket"],
        ["ctx", "innings", "phase"],
        ["ctx", "innings"],
    ]

    for columns in levels:
        grouped = frame.groupby(
            columns,
            dropna=False,
            observed=True,
        )[value]

        means = grouped.transform("mean")
        counts = grouped.transform("count")

        use = (
            out.isna()
            & counts.ge(min_count)
            & means.notna()
        )

        out.loc[use] = means.loc[use]

    return out.fillna(frame[value].mean())


def add_ball_values(bb, lam):
    bb = bb.copy()

    bb["wk_bucket"] = pd.cut(
        bb["wk_before"],
        [-1, 1, 3, 5, 10],
        labels=False,
    )

    if "pace_or_spin" in bb.columns:
        bb["pos"] = (
            bb["pace_or_spin"]
            .fillna("NA")
            .astype(str)
        )
    else:
        bb["pos"] = "NA"

    bb["lam"] = bb["phase"].map(lam).fillna(10)

    bat = bb.loc[~bb["is_wide"]].copy()

    bat_key = [
        "ctx", "innings", "over_idx", "wk_bucket", "pos"
    ]

    bat["expected_bat_runs"] = smoothed_group_mean(
        bat, bat_key, "bat_runs"
    )

    bat["striker_wk"] = (
        bat["wk"].astype(bool)
        & bat["dismissed_player_id"].eq(bat["striker_id"])
    )

    bat["expected_striker_out"] = smoothed_group_mean(
        bat, bat_key, "striker_wk"
    )

    bat["striker_value"] = (
        bat["bat_runs"]
        - bat["expected_bat_runs"]
        + bat["lam"] * bat["expected_striker_out"]
    )

    bb["bat_striker_value"] = np.nan
    bb["bat_striker_id"] = np.nan

    bb.loc[bat.index, "bat_striker_value"] = (
        bat["striker_value"]
    )
    bb.loc[bat.index, "bat_striker_id"] = (
        bat["striker_id"]
    )

    bb["bat_wicket_value"] = np.nan
    bb["bat_wicket_player_id"] = np.nan

    wicket_mask = (
        bb["wk"].astype(bool)
        & bb["dismissed_player_id"].notna()
    )

    bb.loc[wicket_mask, "bat_wicket_value"] = (
        -bb.loc[wicket_mask, "lam"]
    )

    bb.loc[wicket_mask, "bat_wicket_player_id"] = (
        bb.loc[wicket_mask, "dismissed_player_id"]
    )

    bowl_key = [
        "ctx", "innings", "over_idx", "wk_bucket", "pos"
    ]

    bb["expected_bowl_runs"] = smoothed_group_mean(
        bb, bowl_key, "bowl_runs"
    )

    bb["expected_bowler_wk"] = smoothed_group_mean(
        bb, bowl_key, "bowler_wk"
    )

    bb["bowl_val"] = (
        -(bb["bowl_runs"] - bb["expected_bowl_runs"])
        + bb["lam"] * (
            bb["bowler_wk"].astype(float)
            - bb["expected_bowler_wk"]
        )
    )

    return bb


def z(series):
    return (
        series - series.mean()
    ) / (series.std(ddof=0) + 1e-9)


def player_rows(bb, role):
    """One row per (player, delivery). is_ball=1 marks a ball faced / bowled; dismissal-penalty rows have is_ball=0."""
    flag = "high_bat" if role == "bat" else "high_bowl"
    common = ["match_id", "innings", flag, "match_date"]

    if role == "bat":
        striker = bb[common + ["bat_striker_id", "bat_striker_value"]].rename(
            columns={"bat_striker_id": "player_id", "bat_striker_value": "value", flag: "high"})
        striker["is_ball"] = 1

        dismissed = bb[common + ["bat_wicket_player_id", "bat_wicket_value"]].rename(
            columns={"bat_wicket_player_id": "player_id", "bat_wicket_value": "value", flag: "high"})
        dismissed["is_ball"] = 0

        d = pd.concat([striker, dismissed], ignore_index=True)
    else:
        d = bb[common + ["bowler_id", "bowl_val"]].rename(
            columns={"bowler_id": "player_id", "bowl_val": "value", flag: "high"})
        d["is_ball"] = 1

    d = d.dropna(subset=["player_id", "value", "high"])
    d["player_id"] = pd.to_numeric(d["player_id"], errors="coerce")
    d = d.dropna(subset=["player_id"])
    return d


def tier_mix(bb, role, q=0.75):
    """Per player: high-pressure balls by competition tier, plus IPL matches played in the data."""
    d = player_rows(assign_pressure_flags(bb, q), role)
    ctx_of = bb.drop_duplicates("match_id").set_index("match_id")["ctx"]
    d["ctx"] = d["match_id"].map(ctx_of)

    high = d[d["high"]]
    mix = (
        high.pivot_table(index="player_id", columns="ctx", values="is_ball",
                         aggfunc="sum", fill_value=0)
        .reindex(columns=["IPL", "T20I_major", "T20I_assoc"], fill_value=0)
        .rename(columns={"IPL": "pn_ipl", "T20I_major": "pn_major", "T20I_assoc": "pn_assoc"})
    )
    mix["ipl_share"] = mix["pn_ipl"] / mix.sum(axis=1).replace(0, np.nan)

    played = (
        d[d["is_ball"] == 1].groupby(["player_id", "ctx"])["match_id"].nunique()
        .unstack(fill_value=0)
        .reindex(columns=["IPL", "T20I_major", "T20I_assoc"], fill_value=0)
        .rename(columns={"IPL": "m_ipl", "T20I_major": "m_major", "T20I_assoc": "m_assoc"})
    )
    return mix.join(played, how="left")


def estimate_k(bb, role):
    """
    Use a declared fixed shrinkage value.

    Split-half variance estimates were unstable in this dataset,
    particularly for bowlers. The value is tested at k/2 and 2*k.
    """
    d = player_rows(
        assign_pressure_flags(bb, 0.75),
        role,
    )

    high = d[d["high"]]
    noise = high["value"].var() if len(high) else np.nan

    cut = high["match_date"].median() if len(high) else pd.NaT

    if len(high) and pd.notna(cut):
        a = (
            high[high["match_date"] < cut]
            .groupby("player_id")["value"]
            .mean()
        )
        b = (
            high[high["match_date"] >= cut]
            .groupby("player_id")["value"]
            .mean()
        )
        joined = pd.concat(
            [a.rename("a"), b.rename("b")],
            axis=1,
        ).dropna()
        true_var = (
            np.cov(joined["a"], joined["b"])[0, 1]
            if len(joined) >= 3
            else np.nan
        )
        n_players = len(joined)
    else:
        true_var = np.nan
        n_players = 0

    return DEFAULT_K, noise, true_var, n_players


def rate(
    bb,
    role,
    q=0.75,
    weights=(0.55, 0.15, 0.30),
    k=DEFAULT_K,
    min_balls=120,
    min_matches=10,
    min_match_balls=6,
    m0=8,
    names=None,
    scope="all",
    informed=True,
):

    flagged = assign_pressure_flags(bb, q=q)
    d = player_rows(flagged, role)

    if scope == "chase":
        d = d[d["innings"] == 2].copy()

    high = d[d["high"]].copy()
    normal = d[~d["high"]].copy()

    a = high.groupby("player_id").agg(p_n=("is_ball", "sum"), p_sum=("value", "sum"))
    b = normal.groupby("player_id").agg(b_n=("is_ball", "sum"), b_sum=("value", "sum"))

    per_match = (
        high.groupby(["player_id", "match_id"])
        .agg(n=("is_ball", "sum"), s=("value", "sum"))
        .reset_index()
    )

    matches_count = (
        per_match.groupby("player_id")["match_id"]
        .nunique()
        .rename("p_matches")
    )

    qualifying = per_match[
        per_match["n"] >= min_match_balls
    ]

    consistency = qualifying.groupby("player_id").agg(
        q_m=("match_id", "nunique"),
        hits=("s", lambda x: (x > 0).sum()),
    )

    t = (
        a.join(b, how="left")
        .join(matches_count, how="left")
        .join(consistency, how="left")
    )

    t = t.fillna({
        "b_n": 0,
        "b_sum": 0,
        "p_matches": 0,
        "q_m": 0,
        "hits": 0,
    })

    t["p_raw_rating"] = 100 * t["p_sum"] / (t["p_n"] + k)
    t["base_rating"] = 100 * t["b_sum"] / (t["b_n"] + k)
    t["delta"] = t["p_raw_rating"] - t["base_rating"]

    xb = t["b_sum"] / (t["b_n"] + k)
    yb = t["p_sum"] / t["p_n"].clip(lower=1)
    fit = (t["p_n"] >= min_balls) & (t["p_matches"] >= min_matches)
    beta, alpha = 0.0, 0.0
    if informed and fit.sum() >= 10:
        w = np.sqrt(t.loc[fit, "p_n"].values)
        beta = float(np.clip(np.polyfit(xb[fit].values, yb[fit].values, 1, w=w)[0], 0, 1))
        alpha = float(np.average(yb[fit].values - beta * xb[fit].values, weights=w))
    prior = alpha + beta * xb
    t["p_rating"] = 100 * (t["p_sum"] + k * prior) / (t["p_n"] + k)

    t["consistency"] = (
        t["hits"] + 0.5 * m0
    ) / (t["q_m"] + m0)

    t = t[
        (t["p_n"] >= min_balls)
        & (t["p_matches"] >= min_matches)
    ].copy()

    if t.empty:
        return t

    t["score"] = (
        weights[0] * z(t["p_rating"])
        + weights[1] * z(t["delta"])
        + weights[2] * z(t["consistency"])
    )

    if names:
        normalized_names = {
            pd.to_numeric(pid, errors="coerce"): name
            for pid, name in names.items()
        }
        t["name"] = t.index.map(normalized_names)
        t["name"] = t["name"].fillna("")
    else:
        t["name"] = t.index.astype(str)

    t = t.sort_values("score", ascending=False)
    t["rank"] = np.arange(1, len(t) + 1)
    t.attrs["beta"] = beta
    t.attrs["alpha"] = alpha

    return t


def split_half(bb, role, k=DEFAULT_K, min_n=40):
    """Chronological split-half Spearman reliability + does first-half usual form predict second-half pressure form?"""
    d = player_rows(assign_pressure_flags(bb, 0.75), role)
    cut = d["match_date"].median()
    halves = []

    for half in (d[d["match_date"] < cut], d[d["match_date"] >= cut]):
        high, normal = half[half["high"]], half[~half["high"]]
        p = high.groupby("player_id").agg(n=("is_ball", "sum"), s=("value", "sum"))
        b = normal.groupby("player_id").agg(bn=("is_ball", "sum"), bs=("value", "sum"))
        x = p.join(b, how="left").fillna(0)
        x["level"] = 100 * x["s"] / (x["n"] + k)
        x["base"] = 100 * x["bs"] / (x["bn"] + k)
        x["delta"] = x["level"] - x["base"]
        halves.append(x)

    joined = halves[0].join(halves[1], lsuffix="_1", rsuffix="_2", how="inner")
    joined = joined[(joined["n_1"] >= min_n) & (joined["n_2"] >= min_n)]
    if len(joined) < 3:
        return len(joined), np.nan, np.nan, np.nan

    return (
        len(joined),
        joined["level_1"].corr(joined["level_2"], method="spearman"),
        joined["delta_1"].corr(joined["delta_2"], method="spearman"),
        joined["base_1"].corr(joined["level_2"], method="spearman"),
    )


def bootstrap_ranks(bb, role, q, k, ratings, n_boot=500, seed=7, min_match_balls=6, m0=8, w=(0.55, 0.15, 0.30)):
    """
    Resample MATCHES with replacement and recompute every eligible player's rating and rank.
    Gives rank uncertainty: P(top 10), P(top 5), mean / median rank and a 90% rank interval.
    The eligible player set is fixed to the main run's eligible players.
    """
    players = ratings.index.astype(float).values
    pos = pd.Series(np.arange(len(players)), index=players)

    d = player_rows(assign_pressure_flags(bb, q), role)
    d["player_id"] = d["player_id"].astype(float)
    d = d[d["player_id"].isin(pos.index)]
    pm = d.groupby(["player_id", "match_id", "high"]).agg(n=("is_ball", "sum"), s=("value", "sum")).reset_index()

    match_ids = pm["match_id"].unique()
    midx = pd.Series(np.arange(len(match_ids)), index=match_ids)
    mi = pm["match_id"].map(midx).values
    pi = pm["player_id"].map(pos).values.astype(int)
    hi = pm["high"].values.astype(bool)
    n, sv = pm["n"].values.astype(float), pm["s"].values.astype(float)

    cols = {
        "hn": np.where(hi, n, 0.0), "hs": np.where(hi, sv, 0.0),
        "bn": np.where(~hi, n, 0.0), "bs": np.where(~hi, sv, 0.0),
        "qual": (hi & (n >= min_match_balls)).astype(float),
        "hit": (hi & (n >= min_match_balls) & (sv > 0)).astype(float),
    }
    P = len(players)
    rng = np.random.default_rng(seed)
    ranks = np.zeros((n_boot, P))

    def zs(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    for b in range(n_boot):
        wm = np.bincount(rng.integers(0, len(match_ids), len(match_ids)), minlength=len(match_ids)).astype(float)
        wr = wm[mi]
        agg = {c: np.bincount(pi, weights=wr * v, minlength=P) for c, v in cols.items()}
        p_n, p_sum, b_n, b_sum = agg["hn"], agg["hs"], agg["bn"], agg["bs"]
        p_raw = 100 * p_sum / (p_n + k)
        base = 100 * b_sum / (b_n + k)
        delta = p_raw - base
        cons = (agg["hit"] + 0.5 * m0) / (agg["qual"] + m0)
        xb = b_sum / (b_n + k)
        yb = p_sum / np.maximum(p_n, 1)
        ok = p_n > 0
        wgt = np.sqrt(p_n[ok])
        beta = float(np.clip(np.polyfit(xb[ok], yb[ok], 1, w=wgt)[0], 0, 1))
        alpha = float(np.average(yb[ok] - beta * xb[ok], weights=wgt))
        p_rating = 100 * (p_sum + k * (alpha + beta * xb)) / (p_n + k)
        score = w[0] * zs(p_rating) + w[1] * zs(delta) + w[2] * zs(cons)
        ranks[b, np.argsort(-score)] = np.arange(1, P + 1)

    out = pd.DataFrame({
        "name": ratings["name"].values,
        "main_rank": ratings["rank"].values,
        "p_top10": (ranks <= 10).mean(axis=0),
        "p_top5": (ranks <= 5).mean(axis=0),
        "mean_rank": ranks.mean(axis=0),
        "median_rank": np.median(ranks, axis=0),
        "rank_p05": np.percentile(ranks, 5, axis=0),
        "rank_p95": np.percentile(ranks, 95, axis=0),
        "p_n": ratings["p_n"].values,
        "p_matches": ratings["p_matches"].values,
    }, index=ratings.index)
    return out.sort_values(["mean_rank"])


def describe_pressure(bb, flag="high_bat"):
    rows = []
    for (innings, high), group in bb.groupby(["innings", flag]):
        row = {
            "innings": innings,
            "pressure": "HIGH" if high else "normal",
            "deliveries": len(group),
            "mean_over": group["over_idx"].mean() + 1,
            "wkts_down": group["wk_before"].mean(),
            "win_prob_closeness": (group["wp"] - 0.5).abs().mean(),
        }
        if innings == 1:
            row.update({
                "runs_vs_par": (group["runs_before"] - group["par_runs"]).mean(),
                "wkts_last_12_balls": group["recent_wickets"].mean(),
                "runs_last_12_balls": group["recent_runs"].mean(),
            })
        else:
            row.update({
                "mean_leverage": group["leverage"].mean(),
                "runs_needed": group["needed"].mean(),
                "balls_left": group["balls_left"].mean(),
                "req_rate": group["rrr"].mean(),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index(["innings", "pressure"])


def mode_name(series):
    mode = series.dropna().astype(str).mode()
    return mode.iat[0] if not mode.empty else ""


def numeric_name_map(frame, id_col, name_col):
    if id_col not in frame or name_col not in frame:
        return {}

    temp = frame[[id_col, name_col]].dropna().copy()
    temp[id_col] = pd.to_numeric(
        temp[id_col], errors="coerce"
    )
    temp = temp.dropna(subset=[id_col])

    return (
        temp.groupby(id_col)[name_col]
        .agg(mode_name)
        .to_dict()
    )


def build_name_maps(bb):
    batter_names = numeric_name_map(
        bb, "striker_id", "striker_name"
    )

    if (
        "dismissed_player_id" in bb.columns
        and "dismissed_player_name" in bb.columns
    ):
        dismissed_names = numeric_name_map(
            bb, "dismissed_player_id", "dismissed_player_name"
        )
        for pid, name in dismissed_names.items():
            if name and not batter_names.get(pid):
                batter_names[pid] = name

    bowler_names = numeric_name_map(
        bb, "bowler_id", "bowler_name"
    )

    return batter_names, bowler_names


def historical_ipl_teams(mapping):
    team_of = {}

    required = {"comp_name", "player_id", "team_name"}
    if not required.issubset(mapping.columns):
        return team_of

    ipl = mapping[
        mapping["comp_name"].astype(str).str.contains(
            "IPL", case=False, na=False
        )
    ].copy()

    ipl["year"] = pd.to_numeric(
        ipl["comp_name"].astype(str).str.extract(
            r"(\d{4})"
        )[0],
        errors="coerce",
    )

    ipl["player_id"] = pd.to_numeric(
        ipl["player_id"], errors="coerce"
    )

    ipl = ipl.dropna(
        subset=["year", "player_id", "team_name"]
    )

    latest = (
        ipl.sort_values("year")
        .groupby("player_id")
        .tail(1)
    )

    return latest.set_index("player_id").apply(
        lambda r: f"{r['team_name']} ({int(r['year'])})",
        axis=1,
    ).to_dict()


def targeted_score_mismatch_audit(bb_raw):
    """Inspect raw deliveries in the 15 largest score mismatches."""

    audit_ids = [
        16424, 32617, 38795, 16259, 29926,
        40743, 41504, 31286, 24823, 30059,
        30341, 17559, 37554, 40753, 20189
    ]

    cols = [
        "match_id", "innings", "over_ball_num",
        "runs", "extras_type", "extras1_runs",
        "extras_2_runs", "dismissal_type"
    ]
    cols = [c for c in cols if c in bb_raw.columns]

    audit = bb_raw[
        bb_raw["match_id"].isin(audit_ids)
    ].copy()

    print("\n========== TARGETED EXTRAS AUDIT ==========")

    if audit.empty:
        print("No matching delivery records found.")
        print("========== END TARGETED AUDIT ==========")
        return

    for mid in audit_ids:
        m = audit[audit["match_id"] == mid]

        if m.empty:
            continue

        print(f"\n\nMATCH {mid}")

        for inn in sorted(m["innings"].dropna().unique()):
            x = m[m["innings"] == inn].copy()

            print(f"\nINNINGS {inn}")
            print(
                "Raw runs total:",
                x["runs"].fillna(0).sum()
            )

            print("\nExtras breakdown:")

            aggregations = {
                "deliveries": ("runs", "size"),
                "runs_total": ("runs", "sum"),
            }

            if "extras_2_runs" in x.columns:
                aggregations["extras2_total"] = (
                    "extras_2_runs", "sum"
                )

            print(
                x.groupby(
                    "extras_type",
                    dropna=False
                )
                .agg(**aggregations)
                .to_string()
            )

            if "extras_2_runs" in x.columns:
                extra2 = pd.to_numeric(
                    x["extras_2_runs"],
                    errors="coerce"
                ).fillna(0)

                unusual = x[
                    x["extras_type"].notna()
                    | extra2.ne(0)
                ]
            else:
                unusual = x[x["extras_type"].notna()]

            print("\nExtra deliveries:")

            if unusual.empty:
                print("No classified extras deliveries.")
            else:
                print(
                    unusual[cols].to_string(index=False)
                )

    print("\n========== END TARGETED AUDIT ==========")


def _name_tokens(text):
    return re.sub(r"[^a-z\s]", " ", str(text).lower()).split()


def resolve_unavailable_ids(path_candidates, batter_names, bowler_names):
    
    path = next((p for p in path_candidates if os.path.exists(p)), None)
    if path is None:
        print(
            "\nWARNING: unavailable-players file not found "
            f"(looked for: {', '.join(path_candidates)}).\n"
            "         Ranking pool still includes players tied to franchises. "
            "Create the file once IPL 2027 retention lists are official."
        )
        return set()

    df = pd.read_csv(path, comment="#")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.empty:
        print(f"\nWARNING: {path} has no rows; no one extra is excluded.")
        return set()

    ids = set()

    if "player_id" in df.columns:
        direct = pd.to_numeric(df["player_id"], errors="coerce")
        ids |= {float(x) for x in direct.dropna()}
        need_name = df[direct.isna()]
    else:
        need_name = df

    if "name" in need_name.columns:
        full_map, short_map = {}, {}
        for names in (batter_names, bowler_names):
            for pid, nm in names.items():
                tok = _name_tokens(nm)
                if not tok:
                    continue
                full_map.setdefault(" ".join(tok), set()).add(float(pid))
                short_map.setdefault(f"{tok[0][0]} {tok[-1]}", set()).add(float(pid))

        unmatched, ambiguous = [], []
        for nm in need_name["name"].dropna():
            tok = _name_tokens(nm)
            if not tok:
                continue
            hit = full_map.get(" ".join(tok))
            if not hit:
                hit = short_map.get(f"{tok[0][0]} {tok[-1]}")
            if not hit:
                unmatched.append(nm)
            elif len(hit) > 1:
                ambiguous.append((nm, sorted(int(h) for h in hit)))
            else:
                ids |= hit

        if unmatched:
            print("  Names NOT found in the ball-by-ball data (no IPL/T20I history, "
                  "or spelled differently):", unmatched)
        if ambiguous:
            print("  AMBIGUOUS names (add player_id to the CSV to fix):", ambiguous)

    print(f"Unavailable-at-auction file {path}: {len(df)} rows -> {len(ids)} player_ids matched")
    return ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--q",
        type=float,
        default=0.75,
        help="High-pressure quantile; default 0.75",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help="Match-level bootstrap resamples for rank uncertainty (0 to skip); "
             f"default {DEFAULT_BOOTSTRAP}",
    )

    parser.add_argument(
        "--exclude-ids",
        default=DEFAULT_EXCLUDE_IDS,
        help="Comma-separated player_ids to drop from the ranked pool (auction-eligibility screen). "
             "Pressure definition and baselines are unchanged; only the ranking pool shrinks. "
             f"Default: {DEFAULT_EXCLUDE_IDS}. Pass \"\" to exclude nobody.",
    )

    parser.add_argument(
        "--unavailable-file",
        default=DEFAULT_UNAVAILABLE_FILE,
        help="CSV (player_id,name,team) of players retained/contracted by any franchise, "
             "i.e. NOT biddable by RR at the IPL 2027 auction. They are dropped from the "
             f"ranked pool. Default: {DEFAULT_UNAVAILABLE_FILE} (looked up in cwd, then --data).",
    )

    args = parser.parse_args()
    exclude_ids = {float(x) for x in args.exclude_ids.split(",") if x.strip()}

    if not 0.5 < args.q < 1:
        raise ValueError("--q must be between 0.5 and 1")

    os.makedirs(args.out, exist_ok=True)

    print(
        f"Settings: data={args.data!r}; out={args.out!r}; q={args.q}; "
        f"bootstrap={args.bootstrap}; excluded player_ids={sorted(int(x) for x in exclude_ids)}"
    )

    MAIN_WEIGHTS = (0.55, 0.15, 0.30)

    bb_raw, matches, competitions, mapping = load(args.data)

    targeted_score_mismatch_audit(bb_raw)

    diagnostics(bb_raw)

    bb = prepare(bb_raw, matches, competitions)

    allowed_contexts = ["IPL", "T20I_major", "T20I_assoc"]

    before_filter = len(bb)
    bb = bb[bb["ctx"].isin(allowed_contexts)].copy()
    bb = bb.reset_index(drop=True)
    excluded = before_filter - len(bb)

    if bb.empty:
        raise ValueError(
            "No IPL or T20I deliveries found. "
            "Check the competition context labels."
        )

    print("\nCompetition scope: IPL + T20Is only")
    print(f"Excluded OTHER/unsupported deliveries: {excluded:,}")
    print(f"Deliveries retained: {len(bb):,}")
    print(
        "Matches by context:\n",
        bb.groupby("ctx")["match_id"].nunique().to_string(),
    )

    print(
        f"\nDeliveries used: {len(bb):,}; "
        f"matches: {bb['match_id'].nunique():,}"
    )
    print("\nTraining out-of-fold win-probability models...")

    bb = win_prob_and_leverage(bb)

    eps = 1e-4
    valid = bb["wp"].notna()

    if not valid.any():
        raise ValueError(
            "No valid out-of-fold win probabilities were produced."
        )

    log_loss = -np.mean(
        np.where(
            bb.loc[valid, "won"] == 1,
            np.log(
                bb.loc[valid, "wp"].clip(eps, 1 - eps)
            ),
            np.log(
                (1 - bb.loc[valid, "wp"]).clip(eps, 1 - eps)
            ),
        )
    )

    print(
        f"\nWP out-of-fold log-loss: {log_loss:.3f} "
        "(0.693 is the coin-flip reference)"
    )

    for innings in (1, 2):
        g = bb[bb["innings"] == innings]
        late = g[g["over_idx"] >= 15]

        print(
            f"Innings {innings}: "
            f"mean |WP - 0.5| overall="
            f"{(g['wp'] - 0.5).abs().mean():.3f}; "
            f"overs 16–20="
            f"{(late['wp'] - 0.5).abs().mean():.3f}"
        )

    lam = wicket_cost_in_runs(bb)
    print("\nWicket costs:", lam)

    bb = add_ball_values(bb, lam)

    bb = first_innings_pressure(bb)
    bb = assign_pressure_flags(bb, q=args.q)

    for label, flag in (("BATTING", "high_bat"), ("BOWLING", "high_bowl")):
        validation = describe_pressure(bb, flag)
        print(f"\nHigh vs normal pressure -- {label}:\n", validation.round(3).to_string())
        validation.to_csv(os.path.join(args.out, f"pressure_validation_{label.lower()}.csv"))
        print(f"\nShare of deliveries flagged high-pressure ({label}), by innings and phase:")
        print(bb.groupby(["innings", "phase"])[flag].mean().round(3).unstack().to_string())

    chase = bb[bb["innings"] == 2].copy()
    chase["rrr_band"] = pd.cut(chase["rrr"], [-1, 6, 8, 10, 12, 15, 100])
    print("\nChase pressure by required-rate band:")
    print(chase.groupby("rrr_band", observed=False)["high_bat"].agg(["mean", "size"]).round(3).to_string())

    batter_names, bowler_names = build_name_maps(bb)
    team_of = historical_ipl_teams(mapping)

    unavailable = resolve_unavailable_ids(
        [args.unavailable_file, os.path.join(args.data, args.unavailable_file)],
        batter_names,
        bowler_names,
    )
    exclude_ids |= unavailable
    print(f"Total players removed from ranked pool: {len(exclude_ids)}")

    results = {}

    for role, names in (
        ("bat", batter_names),
        ("bowl", bowler_names),
    ):
        print(f"\n========== {role.upper()} ==========")

        k, noise, true_var, n_players = estimate_k(bb, role)

        print(
            f"Fixed shrinkage k={k}; "
            f"high-pressure noise variance={noise}; "
            f"split-half covariance diagnostic={true_var}; "
            f"players in diagnostic={n_players}"
        )

        print(
            "NOTE: k is a declared regularisation setting, "
            "not an estimated player-population parameter."
        )

        ratings = rate(
            bb,
            role,
            q=args.q,
            k=k,
            weights=MAIN_WEIGHTS,
            names=names,
        )

        if exclude_ids:
            drop = ratings.index.astype(float).isin(exclude_ids)
            print(f"Auction-eligibility screen: dropping {int(drop.sum())} player(s): "
                  f"{', '.join(map(str, ratings.loc[drop, 'name'].tolist()))}")
            attrs0 = dict(ratings.attrs)
            ratings = ratings.loc[~drop].copy()
            ratings["rank"] = np.arange(1, len(ratings) + 1)
            ratings.attrs.update(attrs0)

        ratings["latest_ipl_team"] = ratings.index.map(team_of)

        saved_attrs = dict(ratings.attrs)
        mix = tier_mix(bb, role, q=args.q)
        ratings = ratings.join(mix, how="left")
        ratings.attrs.update(saved_attrs)  

        ratings.to_csv(
            os.path.join(args.out, f"{role}_ratings.csv")
        )

        results[role] = ratings

        display_cols = [
            "name",
            "latest_ipl_team",
            "p_n",
            "p_matches",
            "p_rating",
            "p_raw_rating",
            "base_rating",
            "delta",
            "consistency",
            "score",
            "pn_ipl",
            "pn_major",
            "pn_assoc",
            "m_ipl",
        ]

        if len(ratings):
            print("\nTop 15 (pn_* = high-pressure balls by tier; m_ipl = IPL matches in data):")
            print(
                ratings[display_cols]
                .head(15)
                .round(2)
                .to_string()
            )
        else:
            print("\nNo players met the eligibility thresholds.")

        print(
            f"Baseline-informed shrinkage: pressure per-ball prior = {ratings.attrs.get('alpha', 0):.3f} "
            f"+ {ratings.attrs.get('beta', 0):.2f} x usual per-ball form"
        )

        n_split, rho_level, rho_delta, rho_base = split_half(bb, role, k=k)

        print(
            f"\nSplit-half reliability: n={n_split}; level={rho_level:.2f}; "
            f"gap-vs-own-baseline={rho_delta:.2f}; "
            f"first-half usual form -> second-half pressure form={rho_base:.2f}"
        )

        alternatives = {
            "q=0.70": rate(
                bb,
                role,
                q=0.70,
                k=k,
                weights=MAIN_WEIGHTS,
                names=names,
            ),

            "q=0.80": rate(
                bb,
                role,
                q=0.80,
                k=k,
                weights=MAIN_WEIGHTS,
                names=names,
            ),

            "chase-only pressure": rate(
                bb,
                role,
                q=args.q,
                k=k,
                weights=MAIN_WEIGHTS,
                names=names,
                scope="chase",
            ),

            "equal weights": rate(
                bb,
                role,
                q=args.q,
                k=k,
                weights=(1 / 3, 1 / 3, 1 / 3),
                names=names,
            ),

            "no gap term": rate(
                bb,
                role,
                q=args.q,
                k=k,
                weights=(0.65, 0.0, 0.35),
                names=names,
            ),

            "shrink-to-zero": rate(
                bb,
                role,
                q=args.q,
                k=k,
                weights=MAIN_WEIGHTS,
                names=names,
                informed=False,
            ),

            "k/2": rate(
                bb,
                role,
                q=args.q,
                k=max(60, k / 2),
                weights=MAIN_WEIGHTS,
                names=names,
            ),

            "k*2": rate(
                bb,
                role,
                q=args.q,
                k=k * 2,
                weights=MAIN_WEIGHTS,
                names=names,
            ),
        }

        if exclude_ids:
            cleaned = {}
            for label, alt in alternatives.items():
                alt = alt.loc[~alt.index.astype(float).isin(exclude_ids)].copy()
                alt["rank"] = np.arange(1, len(alt) + 1)
                cleaned[label] = alt
            alternatives = cleaned

        base_top = ratings.head(10)
        base_ids = set(base_top.index)

        print("\nTop-10 overlap with main leaderboard:")

        for label, alternative in alternatives.items():
            overlap = len(
                base_ids & set(alternative.head(10).index)
            )
            print(f"  {label}: {overlap}/10")

        stability = pd.DataFrame(
            {
                label: [
                    alt["rank"].get(player, np.nan)
                    for player in base_top.index
                ]
                for label, alt in alternatives.items()
            },
            index=base_top["name"],
        )

        stability.insert(
            0,
            "base",
            range(1, len(base_top) + 1),
        )

        stability.to_csv(
            os.path.join(
                args.out,
                f"{role}_rank_stability.csv",
            )
        )

        print("\nRank stability:")
        print(stability.to_string())

        if args.bootstrap > 0 and len(ratings) >= 20:
            boot = bootstrap_ranks(bb, role, args.q, k, ratings, n_boot=args.bootstrap)
            boot.to_csv(os.path.join(args.out, f"{role}_bootstrap.csv"))
            results[role + "_boot"] = boot
            print(f"\nBOOTSTRAP ({args.bootstrap} match resamples) -- top 20 by mean rank:")
            print(boot.head(20).round(2).to_string())

    print("\n========== CANDIDATE LISTS — PROVISIONAL ==========")
    for role in ("bat", "bowl"):
        if role + "_boot" in results:
            print(f"\n{role.upper()} ordered by bootstrap mean rank:")
            for i, (_, r) in enumerate(results[role + "_boot"].head(10).iterrows(), 1):
                print(f"  {i}. {r['name']}  (main rank {int(r['main_rank'])}, P(top10)={r['p_top10']:.0%})")

    for role in ("bat", "bowl"):
        print(f"\n{role.upper()}:")

        if len(results[role]):
            for i, nm in enumerate(results[role].head(10)["name"], 1):
                print(f"  {i}. {nm}")
        else:
            print("No eligible players.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for innings, flag, lab in (
            (1, "high_bat", "1st innings - batters (behind par / collapse)"),
            (1, "high_bowl", "1st innings - bowlers (ahead of par / attacked)"),
            (2, "high_bat", "Chase - both (win-prob leverage)"),
        ):
            g = bb[bb["innings"] == innings]
            series = g.groupby("over_idx")[flag].mean() * 100
            ax.plot(series.index + 1, series.values, marker="o", label=lab)
        ax.set_xlabel("Over")
        ax.set_ylabel("% of deliveries flagged high-pressure")
        ax.set_title("Where high pressure occurs - IPL + T20Is")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "chart_pressure_by_over.png"), dpi=160)
        plt.close(fig)

        for role, title in (
            ("bat", "Batters"),
            ("bowl", "Bowlers"),
        ):
            t = results[role]

            if t.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.scatter(
                t["base_rating"],
                t["p_rating"],
                s=12,
                alpha=0.4,
            )

            top = t.head(10)

            ax.scatter(
                top["base_rating"],
                top["p_rating"],
                s=28,
            )

            for _, row in top.iterrows():
                ax.annotate(
                    str(row["name"]),
                    (
                        row["base_rating"],
                        row["p_rating"],
                    ),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

            low = min(
                t["base_rating"].min(),
                t["p_rating"].min(),
            )

            high = max(
                t["base_rating"].max(),
                t["p_rating"].max(),
            )

            ax.plot(
                [low, high],
                [low, high],
                "k--",
                lw=0.7,
            )

            ax.set_xlabel(
                "Normal-pressure rating "
                "(runs above average per 100 balls)"
            )

            ax.set_ylabel(
                "High-pressure rating "
                "(runs above average per 100 balls)"
            )

            ax.set_title(
                f"{title}: pressure vs normal performance"
            )

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    args.out,
                    f"chart_{role}_pressure_vs_normal.png",
                ),
                dpi=160,
            )

            plt.close(fig)

    except Exception as error:
        print("Chart generation skipped:", error)

    print(
        f"\nFinished. IPL + T20I results saved in: {args.out}"
    )


if __name__ == "__main__":
    main()