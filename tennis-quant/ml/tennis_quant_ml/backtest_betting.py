from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .betting import evaluate_fixed_threshold_strategy
from .odds import load_odds_csv, merge_predictions_with_odds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--odds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--default-tour", choices=["ATP", "WTA"])
    parser.add_argument("--min-edge", type=float, default=0.04)
    parser.add_argument("--min-ev", type=float, default=0.02)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, low_memory=False)
    odds = load_odds_csv(args.odds, default_tour=args.default_tour)
    matched = merge_predictions_with_odds(predictions, odds)

    report, bets = evaluate_fixed_threshold_strategy(
        matched,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
    )

    report["predictions_total"] = int(len(predictions))
    report["odds_rows_total"] = int(len(odds))
    report["match_rate"] = (
        float(len(matched) / len(predictions))
        if len(predictions)
        else 0.0
    )
    report["warning"] = (
        "Thresholds are fixed before evaluation. Do not tune them on the same "
        "out-of-sample period and then report that period as unbiased."
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bets.to_csv(output / "bets.csv", index=False)
    (output / "betting_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
