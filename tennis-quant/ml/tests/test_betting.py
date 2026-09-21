import pandas as pd

from tennis_quant_ml.betting import evaluate_fixed_threshold_strategy
from tennis_quant_ml.odds import load_odds_csv, merge_predictions_with_odds


def test_odds_alignment_and_roi(tmp_path):
    predictions = pd.DataFrame(
        [
            {
                "match_date": "2025-01-01",
                "tour": "ATP",
                "tournament": "Test Open",
                "player_a": "Alpha Player",
                "player_b": "Beta Player",
                "target": 1,
                "probability_a": 0.70,
                "probability_b": 0.30,
            }
        ]
    )

    odds_path = tmp_path / "odds.csv"
    pd.DataFrame(
        [
            {
                "tour": "ATP",
                "match_date": "2025-01-02",
                "tournament": "Test Open",
                "player_1": "Beta Player",
                "player_2": "Alpha Player",
                "odds_1": 2.50,
                "odds_2": 1.60,
                "closing_odds_1": 2.60,
                "closing_odds_2": 1.50,
            }
        ]
    ).to_csv(odds_path, index=False)

    odds = load_odds_csv(odds_path)
    matched = merge_predictions_with_odds(predictions, odds)

    assert len(matched) == 1
    assert matched.iloc[0]["odds_a"] == 1.60
    assert matched.iloc[0]["odds_b"] == 2.50

    report, bets = evaluate_fixed_threshold_strategy(
        matched,
        min_edge=0.04,
        min_ev=0.02,
    )

    assert report["bets"] == 1
    assert round(report["profit_units"], 6) == 0.6
    assert round(report["roi"], 6) == 0.6
    assert round(report["average_clv"], 6) == round(1.6 / 1.5 - 1.0, 6)
    assert bool(bets.iloc[0]["won"]) is True
