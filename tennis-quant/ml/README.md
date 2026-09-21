# Tennis Quant ML

Leakage-safe research pipeline for ATP/WTA singles.

## What it does

1. Downloads yearly ATP/WTA match files from the archival Sackmann mirror.
2. Filters unfinished/walkover records and non-target event levels.
3. Sorts chronologically using tournament date + explicit round order.
4. Creates **pre-match** features before updating player state.
5. Maintains global Elo and surface Elo.
6. Adds smoothed Hold%, Break%, recent form and 14-day match load.
7. Trains ATP and WTA independently.
8. Fits a regularized logistic baseline.
9. Uses a prior 180-day calibration window with Platt scaling.
10. Evaluates only on future years using walk-forward folds.
11. Saves model artifacts and JSON metrics.

## Important leakage guard

Do **not** sort modern files only by `match_num`. Some 2025+ data reversed that convention. The pipeline uses explicit round ordering first.

## Research data license

The default archive is distributed under **CC BY-NC-SA 4.0** and is therefore for research/non-commercial use. Production data must come from a source whose license permits the intended deployment.

## Run

From `tennis-quant/ml`:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m tennis_quant_ml.train \
  --tour ATP \
  --start-year 2000 \
  --end-year 2026 \
  --test-years 2022 2023 2024 2025 \
  --output artifacts/atp

python -m tennis_quant_ml.train \
  --tour WTA \
  --start-year 2000 \
  --end-year 2026 \
  --test-years 2022 2023 2024 2025 \
  --output artifacts/wta
```

## Betting metrics

ROI and CLV intentionally remain null until timestamped historical odds are supplied. Model accuracy is not evidence of betting profitability.
