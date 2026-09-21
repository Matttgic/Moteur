from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/"
    "Aneeshers/tennis-sackmann-archive/main/{folder}/{prefix}_matches_{year}.csv"
)

ROUND_ORDER = {
    "RR": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}

VALID_LEVELS = {"G", "M", "A", "F", "P"}

REQUIRED_COLUMNS = {
    "tourney_name",
    "surface",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_age",
    "loser_age",
    "score",
    "round",
}


def _source_url(tour: str, year: int) -> str:
    tour = tour.upper()
    if tour not in {"ATP", "WTA"}:
        raise ValueError("tour must be ATP or WTA")
    prefix = tour.lower()
    return ARCHIVE_BASE.format(folder=prefix, prefix=prefix, year=year)


def load_year(tour: str, year: int, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{tour.lower()}_matches_{year}.csv"
    cached = cache_dir / filename

    if not cached.exists():
        urlretrieve(_source_url(tour, year), cached)

    frame = pd.read_csv(cached, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    frame["tour"] = tour.upper()
    return frame


def clean_matches(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    out = out[out["tourney_level"].isin(VALID_LEVELS)]
    out = out[out["surface"].notna()]
    out = out[out["winner_name"].notna() & out["loser_name"].notna()]

    score = out["score"].fillna("").astype(str).str.upper()
    invalid_score = score.str.contains(r"\b(?:W/O|WO|DEF|ABD)\b", regex=True)
    retirement = score.str.contains(r"\bRET\b", regex=True)
    out = out[~invalid_score & ~retirement]

    out["match_date"] = pd.to_datetime(
        out["tourney_date"].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    out = out[out["match_date"].notna()]

    out["round_order"] = out["round"].map(ROUND_ORDER).fillna(-1).astype(int)
    out["match_num_sort"] = pd.to_numeric(out["match_num"], errors="coerce").fillna(0)

    # match_num alone is unsafe for chronology because its convention changed
    # in some 2025+ files. Round order is the primary within-tournament key.
    out = out.sort_values(
        ["match_date", "tourney_name", "round_order", "match_num_sort"],
        kind="stable",
    ).reset_index(drop=True)

    return out


def load_range(
    tour: str,
    start_year: int,
    end_year: int,
    cache_dir: str | Path = ".cache/tennis",
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    cache = Path(cache_dir)
    frames = [load_year(tour, year, cache) for year in range(start_year, end_year + 1)]
    return clean_matches(pd.concat(frames, ignore_index=True))
