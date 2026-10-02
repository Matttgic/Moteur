import pandas as pd

import pytest

from tennis_quant_ml.features import FEATURE_COLUMNS, build_features, export_player_states


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


def test_elo_updates_are_zero_sum():
    date = pd.Timestamp("2025-03-01")
    matches = pd.DataFrame(
        [
            _match(date, 1, "Alpha", "Beta"),
            _match(date + pd.Timedelta(days=7), 2, "Alpha", "Gamma"),
            _match(date + pd.Timedelta(days=14), 3, "Gamma", "Alpha", "Clay"),
        ]
    )

    state = export_player_states(matches, "ATP")
    players = state["players"]

    assert sum(p["elo"] for p in players) == pytest.approx(1500.0 * len(players))
    for surface in ("Hard", "Clay"):
        total = sum(p["surface_elo"][surface] for p in players)
        assert total == pytest.approx(1500.0 * len(players))

    # First match from equal priors: winner +14, loser -14 with K=28.
    beta = next(p for p in players if p["name"] == "Beta")
    assert beta["elo"] == pytest.approx(1486.0)


def test_upset_costs_the_favourite_more_than_an_expected_loss():
    date = pd.Timestamp("2025-04-01")
    warmup = [
        _match(date + pd.Timedelta(days=i), i + 1, "Fav", f"Filler{i}")
        for i in range(6)
    ]
    upset = pd.DataFrame(warmup + [_match(date + pd.Timedelta(days=10), 20, "Dog", "Fav")])
    expected = pd.DataFrame(warmup + [_match(date + pd.Timedelta(days=10), 20, "Fav", "Dog")])

    def elo(frame, name):
        players = export_player_states(frame, "ATP")["players"]
        return next(p["elo"] for p in players if p["name"] == name)

    fav_before = elo(pd.DataFrame(warmup), "Fav")
    fav_loss = fav_before - elo(upset, "Fav")
    dog_loss = 1500.0 - elo(expected, "Dog")

    assert fav_loss > dog_loss > 0
