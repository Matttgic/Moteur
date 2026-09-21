from __future__ import annotations

import numpy as np
import pandas as pd


def no_vig_probabilities(odds_a: float, odds_b: float) -> tuple[float, float]:
    if odds_a <= 1.0 or odds_b <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    inv_a = 1.0 / odds_a
    inv_b = 1.0 / odds_b
    total = inv_a + inv_b
    return inv_a / total, inv_b / total


def _max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    cumulative = profits.cumsum()
    running_max = cumulative.cummax().clip(lower=0.0)
    drawdown = cumulative - running_max
    return float(abs(drawdown.min()))


def evaluate_fixed_threshold_strategy(
    matched: pd.DataFrame,
    min_edge: float = 0.04,
    min_ev: float = 0.02,
) -> tuple[dict, pd.DataFrame]:
    if matched.empty:
        return {
            "matched_matches": 0,
            "bets": 0,
            "profit_units": 0.0,
            "roi": None,
            "hit_rate": None,
            "average_odds": None,
            "average_edge": None,
            "average_ev": None,
            "average_clv": None,
            "max_drawdown_units": 0.0,
        }, matched.copy()

    rows = []
    for _, row in matched.iterrows():
        pa = float(row["probability_a"])
        pb = float(row["probability_b"])
        oa = float(row["odds_a"])
        ob = float(row["odds_b"])

        market_a, market_b = no_vig_probabilities(oa, ob)
        edge_a = pa - market_a
        edge_b = pb - market_b
        ev_a = pa * oa - 1.0
        ev_b = pb * ob - 1.0

        if ev_a >= ev_b:
            side = "A"
            probability = pa
            odds = oa
            edge = edge_a
            ev = ev_a
            won = int(row["target"]) == 1
            closing = row.get("closing_odds_a", np.nan)
        else:
            side = "B"
            probability = pb
            odds = ob
            edge = edge_b
            ev = ev_b
            won = int(row["target"]) == 0
            closing = row.get("closing_odds_b", np.nan)

        if edge < min_edge or ev < min_ev:
            continue

        profit = odds - 1.0 if won else -1.0
        closing_value = np.nan
        if pd.notna(closing) and float(closing) > 1.0:
            closing_value = odds / float(closing) - 1.0

        rows.append(
            {
                **row.to_dict(),
                "bet_side": side,
                "bet_probability": probability,
                "bet_odds": odds,
                "edge": edge,
                "expected_value": ev,
                "won": bool(won),
                "profit_units": profit,
                "clv": closing_value,
            }
        )

    bets = pd.DataFrame(rows)
    if bets.empty:
        return {
            "matched_matches": int(len(matched)),
            "bets": 0,
            "profit_units": 0.0,
            "roi": None,
            "hit_rate": None,
            "average_odds": None,
            "average_edge": None,
            "average_ev": None,
            "average_clv": None,
            "max_drawdown_units": 0.0,
        }, bets

    profit = float(bets["profit_units"].sum())
    count = int(len(bets))
    report = {
        "matched_matches": int(len(matched)),
        "bets": count,
        "profit_units": profit,
        "roi": profit / count,
        "hit_rate": float(bets["won"].mean()),
        "average_odds": float(bets["bet_odds"].mean()),
        "average_edge": float(bets["edge"].mean()),
        "average_ev": float(bets["expected_value"].mean()),
        "average_clv": (
            float(bets["clv"].dropna().mean())
            if bets["clv"].notna().any()
            else None
        ),
        "max_drawdown_units": _max_drawdown(bets["profit_units"]),
        "min_edge": min_edge,
        "min_ev": min_ev,
    }
    return report, bets
