from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_range
from .features import FEATURE_COLUMNS, build_features, export_player_states

MODEL_SPECS = {
    "rank_only_logit": ["log_rank_diff"],
    "elo_logit": ["elo_diff", "surface_elo_diff"],
    "full_logit": FEATURE_COLUMNS,
    "full_hist_gb": FEATURE_COLUMNS,
}

PREDICTION_META_COLUMNS = [
    "match_date",
    "tour",
    "tournament",
    "round",
    "match_num",
    "surface",
    "player_a",
    "player_b",
    "target",
]


def expected_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    if len(y_true) == 0:
        return float("nan")

    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p <= right if right == 1.0 else p < right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _make_estimator(model_name: str):
    if model_name in {"rank_only_logit", "elo_logit", "full_logit"}:
        return Pipeline(
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

    if model_name == "full_hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=35,
            l2_regularization=1.0,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {model_name}")


def _fit_base(train: pd.DataFrame, model_name: str, columns: list[str]):
    model = _make_estimator(model_name)
    model.fit(train[columns], train["target"])
    return model


def _raw_score(base, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if hasattr(base, "decision_function"):
        score = np.asarray(base.decision_function(frame[columns]), dtype=float)
        return score.reshape(-1)

    probability = np.asarray(base.predict_proba(frame[columns])[:, 1], dtype=float)
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _fit_platt(base, calibration: pd.DataFrame, columns: list[str]) -> LogisticRegression:
    score = _raw_score(base, calibration, columns).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=2000, solver="lbfgs")
    calibrator.fit(score, calibration["target"])
    return calibrator


def _predict(base, calibrator, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    score = _raw_score(base, frame, columns).reshape(-1, 1)
    return calibrator.predict_proba(score)[:, 1]


def walk_forward(
    features: pd.DataFrame,
    test_years: list[int],
    model_name: str,
    columns: list[str],
    calibration_days: int = 180,
) -> tuple[list[dict], np.ndarray, np.ndarray, pd.DataFrame]:
    fold_metrics: list[dict] = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_rows: list[pd.DataFrame] = []

    for year in test_years:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year + 1, month=1, day=1)
        calibration_start = test_start - pd.DateOffset(days=calibration_days)

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

        base = _fit_base(train, model_name, columns)
        calibrator = _fit_platt(base, calibration, columns)
        probability = _predict(base, calibrator, test, columns)
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

        prediction_rows = test[PREDICTION_META_COLUMNS].copy()
        prediction_rows["test_year"] = year
        prediction_rows["model_name"] = model_name
        prediction_rows["probability_a"] = probability
        prediction_rows["probability_b"] = 1.0 - probability
        all_rows.append(prediction_rows)

        all_y.append(y)
        all_p.append(probability)

    if not all_y:
        raise RuntimeError("No valid walk-forward fold. Increase historical coverage.")

    return (
        fold_metrics,
        np.concatenate(all_y),
        np.concatenate(all_p),
        pd.concat(all_rows, ignore_index=True),
    )


def evaluate_model(
    features: pd.DataFrame,
    test_years: list[int],
    model_name: str,
    columns: list[str],
) -> tuple[dict, pd.DataFrame]:
    folds, y, probability, predictions = walk_forward(
        features,
        test_years,
        model_name=model_name,
        columns=columns,
    )
    report = {
        "feature_columns": columns,
        "test_n": int(len(y)),
        "aggregate": {
            "log_loss": float(log_loss(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "accuracy": float(accuracy_score(y, probability >= 0.5)),
            "ece_10": expected_calibration_error(y, probability, bins=10),
        },
        "folds": folds,
    }
    return report, predictions


def benchmark_models(
    features: pd.DataFrame,
    test_years: list[int],
) -> tuple[dict[str, dict], str, pd.DataFrame]:
    reports: dict[str, dict] = {}
    predictions: dict[str, pd.DataFrame] = {}

    for model_name, columns in MODEL_SPECS.items():
        report, prediction_rows = evaluate_model(
            features,
            test_years,
            model_name,
            columns,
        )
        reports[model_name] = report
        predictions[model_name] = prediction_rows

    champion = min(
        reports,
        key=lambda name: (
            reports[name]["aggregate"]["log_loss"],
            reports[name]["aggregate"]["brier_score"],
        ),
    )
    return reports, champion, predictions[champion]


def fit_final(
    features: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    columns: list[str],
    tour: str,
    calibration_days: int = 180,
) -> dict:
    latest = pd.Timestamp(features["match_date"].max())
    calibration_start = latest - pd.DateOffset(days=calibration_days)

    train = features[features["match_date"] < calibration_start]
    calibration = features[features["match_date"] >= calibration_start]

    if len(train) < 1000 or len(calibration) < 150:
        raise RuntimeError("Not enough data to fit final calibrated model.")

    base = _fit_base(train, model_name, columns)
    calibrator = _fit_platt(base, calibration, columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, output_dir / "base_model.joblib")
    joblib.dump(calibrator, output_dir / "platt_calibrator.joblib")

    result = {
        "model_name": model_name,
        "trained_through": str(latest.date()),
        "train_n": int(len(train)),
        "calibration_n": int(len(calibration)),
        "feature_columns": columns,
    }

    if model_name.endswith("_logit"):
        scaler = base.named_steps["scale"]
        logit_model = base.named_steps["logit"]
        coefficients = logit_model.coef_[0]

        result["standardized_logit_coefficients"] = {
            feature: float(value) for feature, value in zip(columns, coefficients)
        }

        inference_spec = {
            "schema_version": 1,
            "tour": tour.upper(),
            "model_name": model_name,
            "trained_through": str(latest.date()),
            "feature_columns": columns,
            "scaler": {
                "mean": [float(value) for value in scaler.mean_],
                "scale": [float(value) for value in scaler.scale_],
            },
            "base_logit": {
                "coefficients": [float(value) for value in logit_model.coef_[0]],
                "intercept": float(logit_model.intercept_[0]),
            },
            "platt": {
                "coefficient": float(calibrator.coef_[0][0]),
                "intercept": float(calibrator.intercept_[0]),
            },
        }

        (output_dir / "inference_spec.json").write_text(
            json.dumps(inference_spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["inference_spec_file"] = "inference_spec.json"

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", required=True, choices=["ATP", "WTA"])
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
    )
    parser.add_argument("--cache-dir", default=".cache/tennis")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = load_range(args.tour, args.start_year, args.end_year, args.cache_dir)
    features = build_features(raw, args.tour)
    state_seed = export_player_states(raw, args.tour)

    model_reports, champion, champion_predictions = benchmark_models(
        features,
        args.test_years,
    )
    champion_report = model_reports[champion]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    champion_predictions.to_csv(output_dir / "oos_predictions.csv", index=False)
    (output_dir / "player_state_seed.json").write_text(
        json.dumps(state_seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    final = fit_final(
        features,
        output_dir,
        model_name=champion,
        columns=MODEL_SPECS[champion],
        tour=args.tour,
    )

    fallback_dir = output_dir / "rank_only_fallback"
    fallback_final = fit_final(
        features,
        fallback_dir,
        model_name="rank_only_logit",
        columns=MODEL_SPECS["rank_only_logit"],
        tour=args.tour,
    )

    report = {
        "tour": args.tour,
        "source": "Aneeshers/tennis-sackmann-archive",
        "source_license": "CC BY-NC-SA 4.0 (research/non-commercial)",
        "rows_raw": int(len(raw)),
        "rows_features": int(len(features)),
        "test_years": args.test_years,
        "champion": champion,
        "aggregate": champion_report["aggregate"],
        "models": model_reports,
        "oos_predictions_file": "oos_predictions.csv",
        "player_state_seed_file": "player_state_seed.json",
        "final_model": final,
        "fallback_model": fallback_final,
        "betting_metrics": {
            "roi": None,
            "clv": None,
            "note": "Not computed without timestamped historical bookmaker odds.",
        },
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
