# Tennis Quant Engine

ATP/WTA research engine for **probability → fair odds → no-vig market probability → edge → expected value → BET / NO BET**.

## V1 goals

- ATP and WTA singles only.
- Transparent baseline model before ML training.
- Surface Elo, global Elo, Hold%, Break%, form, fatigue and physical-risk inputs.
- Conservative uncertainty shrinkage toward 50%.
- No-vig bookmaker comparison.
- Edge / EV filters with explicit NO BET.
- Stakes capped at 0.75u before validated backtests.
- Brier Score, Log Loss, ROI and CLV utilities.
- Supabase schema designed for time-safe historical snapshots.
- Mobile-first dashboard.

## Important status

The included dashboard uses **illustrative demo players and odds**. It is not a live betting feed.

The V0.1 weights in `lib/model.ts` are deliberately transparent placeholders. They must be replaced or validated by leakage-safe walk-forward backtesting before any performance claims are made.

## Local setup

```bash
npm install
npm run typecheck
npm run build
npm run dev
```

Copy `.env.example` to `.env.local` when a dedicated Supabase project is ready.

## API

### POST /api/predict

Input:

```json
{
  "tour": "ATP",
  "surface": "Hard",
  "playerA": {
    "name": "A",
    "elo": 2010,
    "surfaceElo": 2030,
    "holdPct": 85,
    "breakPct": 24,
    "formScore": 7.5,
    "fatigueScore": 1.5,
    "physicalRisk": 0.05
  },
  "playerB": {
    "name": "B",
    "elo": 1940,
    "surfaceElo": 1920,
    "holdPct": 82,
    "breakPct": 22,
    "formScore": 6.4,
    "fatigueScore": 2.0,
    "physicalRisk": 0.08
  },
  "oddsA": 1.65,
  "oddsB": 2.25,
  "dataQuality": 0.9
}
```

## Model governance roadmap

1. Ingest historical ATP/WTA results chronologically.
2. Build pre-match snapshots only from information available at prediction time.
3. Add global Elo + surface Elo + decayed form + serve/return features.
4. Train ATP and WTA separately.
5. Compare logistic regression, gradient boosting and calibrated ensemble.
6. Use walk-forward validation only.
7. Track Brier, Log Loss, calibration, ROI, drawdown and CLV.
8. Segment by tour, surface and odds band.
9. Promote learned weights only when out-of-sample performance is stable.
10. Keep NO BET as a first-class output.

## Supabase security

The migration enables RLS on every public table and creates no permissive client policies. Use a server-side service-role key only in trusted server code; never expose it in `NEXT_PUBLIC_*` variables.
