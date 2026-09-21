from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_range
from .features import FEATURE_COLUMNS, build_features


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    if total == 0:
        return float("nan")

    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _fit_base(train: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    return model


def _fit_platt(base: Pipeline, calibration: pd.DataFrame) -> LogisticRegression:
    decision = base.decision_function(calibration[FEATURE_COLUMNS]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=2000, solver="lbfgs")
    calibrator.fit(decision, calibration["target"])
    return calibrator


def _predict(base: Pipeline, calibrator: LogisticRegression, frame: pd.DataFrame) -> np.ndarray:
    decision = base.decision_function(frame[FEATURE_COLUMNS]).reshape(-1, 1)
    return calibrator.predict_proba(decision)[:, 1]


def walk_forward(
    features: pd.DataFrame,
    test_years: list[int],
    calibration_days: int = 180,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    fold_metrics: list[dict] = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []

    for year in test_years:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year + 1, month=1, day=1)
        calibration_start = test_start - pd.Timedelta(days=calibration_days)

        train = features[features["match_date"] < calibration_start]
        calibration = features[
            (features["match_date"] >= calibration_start)
            & (features["match_date"] < test_start)
        ]
        test = features[
            (features["match_date"] >= test_start)
            & (features["match_date"] < test_end)
        ]

        if len(train) < 1000 or len(calibration) < 150 or len(test) < 100:
            continue

        base = _fit_base(train)
        calibrator = _fit_platt(base, calibration)
        probability = _predict(base, calibrator, test)
        y = test["target"].to_numpy(dtype=int)

        fold_metrics.append(
            {
                "year": year,
                "train_n": int(len(train)),
                "calibration_n": int(len(calibration)),
                "test_n": int(len(test)),
                "log_loss": float(log_loss(y, probability)),
                "brier_score": float(brier_score_loss(y, probability)),
                "accuracy": float(accuracy_score(y, probability >= 0.5)),
                "ece_10": expected_calibration_error(y, probability, bins=10),
            }
        )
        all_y.append(y)
        all_p.append(probability)

    if not all_y:
        raise RuntimeError("No valid walk-forward fold. Increase historical coverage.")

    return fold_metrics, np.concatenate(all_y), np.concatenate(all_p)


def fit_final(features: pd.DataFrame, output_dir: Path, calibration_days: int = 180) -> dict:
    latest = pd.Timestamp(features["match_date"].max())
    calibration_start = latest - pd.Timedelta(days=calibration_days)

    train = features[features["match_date"] < calibration_start]
    calibration = features[features["match_date"] >= calibration_start]

    if len(train) < 1000 or len(calibration) < 150:
        raise RuntimeError("Not enough data to fit final calibrated model.")

    base = _fit_base(train)
    calibrator = _fit_platt(base, calibration)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, output_dir / "base_model.joblib")
    joblib.dump(calibrator, output_dir / "platt_calibrator.joblib")

    coefficients = base.named_steps["logit"].coef_[0]
    return {
        "trained_through": str(latest.date()),
        "train_n": int(len(train)),
        "calibration_n": int(len(calibration)),
        "feature_columns": FEATURE_COLUMNS,
        "standardized_logit_coefficients": {
            feature: float(value) for feature, value in zip(FEATURE_COLUMNS, coefficients)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", required=True, choices=["ATP", "WTA"])
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--test-years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--cache-dir", default=".cache/tennis")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = load_range(args.tour, args.start_year, args.end_year, args.cache_dir)
    features = build_features(raw, args.tour)

    folds, y, probability = walk_forward(features, args.test_years)
    output_dir = Path(args.output)
    final = fit_final(features, output_dir)

    report = {
        "tour": args.tour,
        "source": "Aneeshers/tennis-sackmann-archive",
        "source_license": "CC BY-NC-SA 4.0 (research/non-commercial)",
        "rows_raw": int(len(raw)),
        "rows_features": int(len(features)),
        "test_n": int(len(y)),
        "aggregate": {
            "log_loss": float(log_loss(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "accuracy": float(accuracy_score(y, probability >= 0.5)),
            "ece_10": expected_calibration_error(y, probability, bins=10),
        },
        "folds": folds,
        "final_model": final,
        "betting_metrics": {
            "roi": None,
            "clv": None,
            "note": "Not computed without timestamped historical bookmaker odds.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
