# Tennis Quant Engine — Production Status

## Ready

- GitHub main contains the validated ATP/WTA quant engine.
- Supabase project `Tennis Quant` is active in `eu-west-3`.
- 7 core tennis tables are deployed with RLS enabled.
- Direct table privileges are revoked from `anon` and `authenticated`.
- Foreign-key indexes are deployed.
- ATP and WTA champion model runs are stored in `tennis_model_runs`.
- Supabase TypeScript types are generated and committed.
- Edge Function `tennis-model-status` is deployed and JWT-protected.
- CI validates:
  - TypeScript
  - Next.js build
  - ML unit tests
  - ATP walk-forward benchmark
  - WTA walk-forward benchmark
  - odds / ROI / CLV utilities

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

## External gates

### Live ATP/WTA fixtures
The app is coded to support a legitimate provider and fail closed when not configured.
A free provider account/key is still required for full ATP + WTA coverage.

Recommended currently verified option:
- Live Tennis API FREE: upcoming/live fixtures, ATP/WTA, no card.
- Configure server-side as `LIVE_TENNIS_API_KEY`.

### Betting odds
Do not scrape bookmaker pages.

Free-key options verified during R&D include:
- Odds API providers with free quotas for ATP/WTA pre-match odds.
- Historical odds remain provider-dependent.

The economic layer must not claim ROI until authorized timestamped historical odds
have been connected and evaluated out of sample.

## Deployment

The code is deployable as a Next.js app from `tennis-quant/`.
The current chat integrations did not expose a safe way to bind the GitHub subdirectory
to a new Vercel project automatically. Do not repurpose an existing Vercel football project.

A Netlify site named `tennis-quant-engine` was created, but the connected deploy tool
requires a source-directory CLI upload. The execution environment in this chat had no
outbound network access, so that upload could not be completed here.

## Non-negotiable behavior

- No invented fixtures.
- No invented odds.
- No forced bets.
- `NO BET` is a valid daily output.
- No profitability claim before economic validation.
