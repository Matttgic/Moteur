from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

STANDARD_REQUIRED = {
    "tour",
    "match_date",
    "tournament",
    "player_1",
    "player_2",
    "odds_1",
    "odds_2",
}


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", text)


def _canonical_pair(player_1: object, player_2: object) -> tuple[str, str]:
    names = sorted([_normalize_text(player_1), _normalize_text(player_2)])
    return names[0], names[1]


def make_match_key(
    tour: object,
    year: int,
    tournament: object,
    player_1: object,
    player_2: object,
) -> str:
    p_low, p_high = _canonical_pair(player_1, player_2)
    return "|".join(
        [
            str(tour).upper().strip(),
            str(int(year)),
            _normalize_text(tournament),
            p_low,
            p_high,
        ]
    )


def _legacy_to_standard(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    if {"Winner", "Loser"}.issubset(out.columns):
        out = out.rename(columns={"Winner": "player_1", "Loser": "player_2"})

        price_pairs = [
            ("AvgW", "AvgL"),
            ("MaxW", "MaxL"),
            ("B365W", "B365L"),
            ("PSW", "PSL"),
        ]
        chosen = next(
            ((w, l) for w, l in price_pairs if w in out.columns and l in out.columns),
            None,
        )
        if chosen is None:
            raise ValueError("Legacy odds file has no supported winner/loser price pair.")

        out = out.rename(columns={chosen[0]: "odds_1", chosen[1]: "odds_2"})

    rename = {
        "Date": "match_date",
        "Tournament": "tournament",
        "Tour": "tour",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def load_odds_csv(path: str | Path, default_tour: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = _legacy_to_standard(frame)

    if "tour" not in frame.columns and default_tour:
        frame["tour"] = default_tour.upper()

    missing = STANDARD_REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Odds file is missing columns: {sorted(missing)}")

    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame = frame[frame["match_date"].notna()].copy()

    for column in ["odds_1", "odds_2", "closing_odds_1", "closing_odds_2"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[(frame["odds_1"] > 1.0) & (frame["odds_2"] > 1.0)].copy()
    frame["year"] = frame["match_date"].dt.year
    frame["player_1_norm"] = frame["player_1"].map(_normalize_text)
    frame["player_2_norm"] = frame["player_2"].map(_normalize_text)
    frame["match_key"] = frame.apply(
        lambda row: make_match_key(
            row["tour"],
            row["year"],
            row["tournament"],
            row["player_1"],
            row["player_2"],
        ),
        axis=1,
    )

    if "bookmaker" not in frame.columns:
        frame["bookmaker"] = "unknown"

    return frame


def prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_date",
        "tour",
        "tournament",
        "player_a",
        "player_b",
        "target",
        "probability_a",
        "probability_b",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    out = predictions.copy()
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out = out[out["match_date"].notna()].copy()
    out["year"] = out["match_date"].dt.year
    out["player_a_norm"] = out["player_a"].map(_normalize_text)
    out["player_b_norm"] = out["player_b"].map(_normalize_text)
    out["match_key"] = out.apply(
        lambda row: make_match_key(
            row["tour"],
            row["year"],
            row["tournament"],
            row["player_a"],
            row["player_b"],
        ),
        axis=1,
    )
    return out


def merge_predictions_with_odds(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    preds = prepare_predictions(predictions)

    if odds["match_key"].duplicated().any():
        # Multiple bookmakers/snapshots can exist. Keep the last row supplied by
        # the caller; production ingestion should preselect a timestamp/provider.
        odds = odds.drop_duplicates("match_key", keep="last")

    merged = preds.merge(
        odds,
        on="match_key",
        how="inner",
        suffixes=("_pred", "_odds"),
        validate="one_to_one",
    )

    if merged.empty:
        return merged

    player_a_is_1 = merged["player_a_norm"] == merged["player_1_norm"]
    player_a_is_2 = merged["player_a_norm"] == merged["player_2_norm"]
    if not (player_a_is_1 | player_a_is_2).all():
        raise ValueError("Player alignment failed after match-key merge.")

    merged["odds_a"] = merged["odds_1"].where(player_a_is_1, merged["odds_2"])
    merged["odds_b"] = merged["odds_2"].where(player_a_is_1, merged["odds_1"])

    if {"closing_odds_1", "closing_odds_2"}.issubset(merged.columns):
        merged["closing_odds_a"] = merged["closing_odds_1"].where(
            player_a_is_1,
            merged["closing_odds_2"],
        )
        merged["closing_odds_b"] = merged["closing_odds_2"].where(
            player_a_is_1,
            merged["closing_odds_1"],
        )

    return merged
