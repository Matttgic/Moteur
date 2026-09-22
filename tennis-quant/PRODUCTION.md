# Tennis Quant Engine — Production Status

## Stack

- **Vercel project:** `moteur`
- **Application root:** `tennis-quant/`
- **Primary function region:** `fra1`
- **Supabase project:** `Tennis Quant` (`eu-west-3`)
- **Primary branch:** `main`

## Production data flow

1. Authenticated server jobs call Live Tennis API and OddsPapi.
2. The calibrated ATP/WTA model computes fair probabilities and betting decisions.
3. Actionable FULL-ML selections are recorded in the economic log.
4. Shadow variants are recorded separately for research.
5. The latest complete ATP/WTA response is persisted in `tennis_selection_snapshots`.
6. Public users read only that snapshot. Public page loads do not consume Live Tennis or odds-provider calls.

## Provider responsibilities

### Live Tennis API BASIC

Used for:

- upcoming ATP/WTA fixtures,
- player/ranking fields carried by provider fixtures,
- completed match history,
- official-result synchronization,
- player-state freshness.

It is **not** the execution-odds source.

### OddsPapi

Primary source for bookmaker moneyline prices.

Caching policy:

- discovery: 15 minutes,
- execution prices: 5 minutes.

### The Odds API

Optional fallback if OddsPapi cannot produce a usable board.

Environment variable:

`THE_ODDS_API_KEY`

If this key is not configured, the system remains operational with OddsPapi but can expose `DATA_UNAVAILABLE` for a tour whose primary board is empty. The UI must never label that condition as `NO BET`.

## Security / quota hardening

- `/api/selections/today` is public but read-only.
- Provider refresh requires `refresh=1` plus `Authorization: Bearer <CRON_SECRET>`.
- Public requests cannot record economic or shadow picks.
- `/api/odds/tennis` is private.
- Requested execution bookmakers are restricted to the French allowlist.
- `SUPABASE_SERVICE_ROLE_KEY`, `CRON_SECRET`, Live Tennis and odds keys remain server-side.
- `tennis_selection_snapshots` has RLS enabled and direct privileges revoked from `anon` and `authenticated`.

## Validated model quality

### ATP

- OOS matches: 10,514
- Accuracy: 65.01%
- Log Loss: 0.61894
- Brier: 0.21546
- ECE-10: 0.00956

### WTA

- OOS matches: 4,485
- Accuracy: 66.06%
- Log Loss: 0.61341
- Brier: 0.21260
- ECE-10: 0.01075

The API reads the `trained_through` field from each committed model specification. It must not hard-code a training date.

## Operations

Vercel cron remains a low-frequency backstop. More frequent production operations can run through Supabase Cron, authenticated with the scheduler token stored in Vault.

Recommended cadence after the hardened build is live:

- ATP selection snapshot: every 30 minutes.
- WTA selection snapshot: every 30 minutes, staggered from ATP.
- Incremental ATP result sync: every 2 hours with a 12-hour window.
- Incremental WTA result sync: every 2 hours, staggered.
- Economic settlement: hourly.
- CLV refresh: hourly.
- Provider health: every 6 hours.

This cadence keeps the public UI fresh without allowing page traffic to consume the Live Tennis BASIC quota.

## Economic validation semantics

- `stakeUnits` in aggregate ROI represents **settled stake**.
- Pending stake is displayed separately from settled stake.
- ROI and profit are realized metrics only.
- A pending pick can settle only after the official result has been synchronized into `tennis_matches`.

## Non-negotiable behavior

- No invented fixtures.
- No invented odds.
- No forced bets.
- `NO BET` means data was available and no selection passed the betting rules.
- `DATA_UNAVAILABLE` means the required market data was not available.
- No claim of guaranteed profitability.
