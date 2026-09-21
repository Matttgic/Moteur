import pandas as pd

from tennis_quant_ml.features import FEATURE_COLUMNS, build_features


def _match(date, num, winner, loser, surface="Hard"):
    return {
        "tourney_date": int(date.strftime("%Y%m%d")),
        "tourney_name": "Test Event",
        "tourney_level": "A",
        "surface": surface,
        "match_num": num,
        "round": "R32" if num < 3 else "R16",
        "match_date": date,
        "winner_name": winner,
        "loser_name": loser,
        "winner_rank": 20,
        "loser_rank": 40,
        "winner_age": 25.0,
        "loser_age": 27.0,
        "w_SvGms": 10,
        "w_bpFaced": 3,
        "w_bpSaved": 2,
        "l_SvGms": 9,
        "l_bpFaced": 5,
        "l_bpSaved": 3,
    }


def test_features_are_pre_match_and_finite():
    date = pd.Timestamp("2025-01-01")
    matches = pd.DataFrame(
        [
            _match(date, 1, "Alpha", "Beta"),
            _match(date + pd.Timedelta(days=7), 2, "Alpha", "Gamma"),
            _match(date + pd.Timedelta(days=14), 3, "Beta", "Gamma"),
        ]
    )

    features = build_features(matches, "ATP")

    assert len(features) == 3
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features[FEATURE_COLUMNS].notna().all().all()

    # First-ever match begins from equal Elo priors; later rows can use prior results.
    assert features.iloc[0]["elo_diff"] == 0.0
    assert (features.iloc[1:]["elo_diff"].abs() > 0).any()


def test_orientation_produces_binary_target():
    date = pd.Timestamp("2025-02-01")
    matches = pd.DataFrame(
        [
            _match(date + pd.Timedelta(days=i), i + 10, f"W{i}", f"L{i}")
            for i in range(20)
        ]
    )
    features = build_features(matches, "WTA")
    assert set(features["target"].unique()).issubset({0, 1})
    assert features["target"].nunique() == 2
