from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from hashlib import sha1
from math import log
from typing import Deque

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "elo_diff",
    "surface_elo_diff",
    "log_rank_diff",
    "age_diff",
    "hold_diff",
    "break_diff",
    "form10_diff",
    "load14_diff",
    "surface_hard",
    "surface_clay",
    "surface_grass",
]

SURFACES = ("Hard", "Clay", "Grass", "Carpet")


@dataclass
class PlayerState:
    elo: float = 1500.0
    surface_elo: dict[str, float] = field(
        default_factory=lambda: {surface: 1500.0 for surface in SURFACES}
    )
    recent_results: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    recent_dates: Deque[pd.Timestamp] = field(default_factory=lambda: deque(maxlen=30))
    service_games: float = 0.0
    service_holds: float = 0.0
    return_games: float = 0.0
    return_breaks: float = 0.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _smoothed_rate(
    successes: float,
    trials: float,
    prior: float,
    strength: float = 24.0,
) -> float:
    return (successes + prior * strength) / (trials + strength)


def _form(state: PlayerState) -> float:
    if not state.recent_results:
        return 0.5
    wins = sum(state.recent_results)
    return (wins + 2.0) / (len(state.recent_results) + 4.0)


def _load14(state: PlayerState, now: pd.Timestamp) -> float:
    cutoff = now - pd.DateOffset(days=14)
    return float(sum(date >= cutoff for date in state.recent_dates))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _rank(value) -> float:
    rank = _safe_float(value, 1000.0)
    return max(1.0, rank)


def _orientation_key(row: pd.Series) -> bool:
    # Match metadata only; winner identity is intentionally excluded.
    raw = "|".join(
        [
            str(row.get("tourney_date", "")),
            str(row.get("tourney_name", "")),
            str(row.get("round", "")),
            str(row.get("match_num", "")),
        ]
    )
    return int(sha1(raw.encode("utf-8")).hexdigest()[-2:], 16) % 2 == 0


def _service_summary(row: pd.Series, prefix: str) -> tuple[float, float]:
    service_games = max(0.0, _safe_float(row.get(f"{prefix}_SvGms")))
    bp_faced = max(0.0, _safe_float(row.get(f"{prefix}_bpFaced")))
    bp_saved = max(0.0, _safe_float(row.get(f"{prefix}_bpSaved")))
    breaks_conceded = min(service_games, max(0.0, bp_faced - bp_saved))
    holds = max(0.0, service_games - breaks_conceded)
    return service_games, holds


def build_features(matches: pd.DataFrame, tour: str) -> pd.DataFrame:
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    rows: list[dict] = []

    hold_prior = 0.80 if tour.upper() == "ATP" else 0.72
    break_prior = 1.0 - hold_prior

    for _, match in matches.iterrows():
        winner = str(match["winner_name"])
        loser = str(match["loser_name"])
        date = pd.Timestamp(match["match_date"])
        surface = str(match["surface"])
        surface_key = surface if surface in SURFACES else "Hard"

        ws = states[winner]
        ls = states[loser]

        winner_snapshot = {
            "name": winner,
            "elo": ws.elo,
            "surface_elo": ws.surface_elo[surface_key],
            "rank": _rank(match.get("winner_rank")),
            "age": _safe_float(match.get("winner_age"), 27.0),
            "hold": _smoothed_rate(ws.service_holds, ws.service_games, hold_prior),
            "break": _smoothed_rate(ws.return_breaks, ws.return_games, break_prior),
            "form10": _form(ws),
            "load14": _load14(ws, date),
        }
        loser_snapshot = {
            "name": loser,
            "elo": ls.elo,
            "surface_elo": ls.surface_elo[surface_key],
            "rank": _rank(match.get("loser_rank")),
            "age": _safe_float(match.get("loser_age"), 27.0),
            "hold": _smoothed_rate(ls.service_holds, ls.service_games, hold_prior),
            "break": _smoothed_rate(ls.return_breaks, ls.return_games, break_prior),
            "form10": _form(ls),
            "load14": _load14(ls, date),
        }

        winner_is_a = _orientation_key(match)
        a = winner_snapshot if winner_is_a else loser_snapshot
        b = loser_snapshot if winner_is_a else winner_snapshot

        rows.append(
            {
                "match_date": date,
                "tour": tour.upper(),
                "tournament": str(match.get("tourney_name", "")),
                "round": str(match.get("round", "")),
                "match_num": str(match.get("match_num", "")),
                "surface": surface,
                "player_a": a["name"],
                "player_b": b["name"],
                "target": 1 if winner_is_a else 0,
                "elo_diff": a["elo"] - b["elo"],
                "surface_elo_diff": a["surface_elo"] - b["surface_elo"],
                "log_rank_diff": log(a["rank"] + 1.0) - log(b["rank"] + 1.0),
                "age_diff": a["age"] - b["age"],
                "hold_diff": a["hold"] - b["hold"],
                "break_diff": a["break"] - b["break"],
                "form10_diff": a["form10"] - b["form10"],
                "load14_diff": a["load14"] - b["load14"],
                "surface_hard": 1.0 if surface == "Hard" else 0.0,
                "surface_clay": 1.0 if surface == "Clay" else 0.0,
                "surface_grass": 1.0 if surface == "Grass" else 0.0,
            }
        )

        global_expected = _expected(ws.elo, ls.elo)
        k_global = 28.0
        ws.elo += k_global * (1.0 - global_expected)
        ls.elo += k_global * (0.0 - global_expected)

        surface_expected = _expected(
            ws.surface_elo[surface_key], ls.surface_elo[surface_key]
        )
        k_surface = 32.0
        ws.surface_elo[surface_key] += k_surface * (1.0 - surface_expected)
        ls.surface_elo[surface_key] += k_surface * (0.0 - surface_expected)

        ws.recent_results.append(1)
        ls.recent_results.append(0)
        ws.recent_dates.append(date)
        ls.recent_dates.append(date)

        w_games, w_holds = _service_summary(match, "w")
        l_games, l_holds = _service_summary(match, "l")
        w_breaks = max(0.0, l_games - l_holds)
        l_breaks = max(0.0, w_games - w_holds)

        ws.service_games += w_games
        ws.service_holds += w_holds
        ws.return_games += l_games
        ws.return_breaks += w_breaks

        ls.service_games += l_games
        ls.service_holds += l_holds
        ls.return_games += w_games
        ls.return_breaks += l_breaks

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    numeric = result[FEATURE_COLUMNS]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature detected")

    return result
