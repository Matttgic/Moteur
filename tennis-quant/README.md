# Tennis Quant Engine

ATP/WTA quantitative research engine for:

**probability → fair odds → no-vig market probability → edge → expected value → BET / NO BET**

## Current production architecture

- **Live Tennis API BASIC**: ATP/WTA fixtures, rankings present in fixture payloads, completed results and historical match sync.
- **OddsPapi**: primary pre-match execution odds source.
- **The Odds API**: optional fallback when OddsPapi has no usable board for a tour.
- **Supabase**: historical tennis state, economic tracking, shadow tracking, health heartbeats and server-only selection snapshots.
- **Vercel / Next.js**: public dashboard and private cron endpoints.

The public dashboard **does not call Live Tennis API or OddsPapi directly**. It reads the latest persisted ATP/WTA selection snapshot. Provider calls and bet recording are reserved for authenticated server jobs.

## Betting policy

- ATP and WTA main-tour singles only.
- No invented fixtures or odds.
- No forced bets.
- Rank-only fallback is informational and can never become an actionable bet.
- Actionable picks require the full calibrated model.
- Execution bookmakers are restricted to the configured French bookmaker allowlist.
- Pinnacle is used only as a sharp reference when available.
- Market outlier guards can block a pick even when raw model edge is positive.
- Profitability is measured from actually recorded picks, not theoretical backtest-only ROI.

## Data freshness

- Live Tennis upcoming fixtures are cached server-side.
- OddsPapi discovery is cached for 15 minutes.
- OddsPapi execution prices are cached for 5 minutes.
- The Odds API fallback odds are cached for 15 minutes.
- Selection snapshots are refreshed only by authenticated jobs.
- The public UI exposes snapshot age and marks stale data.

## Economic validation

The dashboard tracks:

- settled and pending bets,
- settled stake vs pending stake,
- net profit,
- realized ROI,
- hit rate,
- average odds,
- maximum drawdown,
- CLV,
- complete recorded pick history.

A pending pick is not counted in realized ROI until an official completed result is synchronized.

## Model validation

Current validated out-of-sample benchmark:

| Tour | OOS matches | Accuracy | Log Loss | Brier | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ATP | 10,514 | 65.01% | 0.61894 | 0.21546 | 0.00956 |
| WTA | 4,485 | 66.06% | 0.61341 | 0.21260 | 0.01075 |

The live API response reports the actual `trained_through` value stored in the model specification instead of a hard-coded date.

## Environment

Server-side secrets:

```bash
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=
SUPABASE_SERVICE_ROLE_KEY=
CRON_SECRET=
LIVE_TENNIS_API_KEY=
ODDS_PAPI_API_KEY=
THE_ODDS_API_KEY=
```

`THE_ODDS_API_KEY` is optional for the primary OddsPapi path but required for The Odds API fallback.

Never expose `SUPABASE_SERVICE_ROLE_KEY`, `CRON_SECRET`, provider API keys, or other server credentials in `NEXT_PUBLIC_*` variables.

## Local setup

```bash
npm install
npm run typecheck
npm run build
npm run dev
```

## Important routes

- `GET /api/selections/today?tour=atp|wta`
  - public, read-only;
  - returns the latest persisted selection snapshot;
  - never refreshes providers or records bets.
- `GET /api/cron/tennis-picks/[tour]`
  - private;
  - refreshes the live board, computes selections, records economic/shadow picks and persists the public snapshot.
- `GET /api/cron/tennis-sync/[tour]?hours=12`
  - private;
  - incrementally synchronizes completed matches and player states.
- `GET /api/cron/tennis-settle`
  - private;
  - settles recorded bets from synchronized official results.
- `GET /api/odds/tennis`
  - private diagnostic endpoint;
  - protected by `CRON_SECRET`.

## Supabase security

All public-schema tennis tables use RLS. The selection snapshot table has no direct `anon` or `authenticated` privileges; the browser receives its data only through trusted server routes using the service-role client.

## Disclaimer

This project is a statistical research and tracking tool. Probabilities, edge and expected value are estimates and do not guarantee profit.
