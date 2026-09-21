create table if not exists public.tennis_shadow_log (
  id uuid primary key default gen_random_uuid(),
  provider_match_id text not null,
  tour text not null check (tour in ('ATP','WTA')),
  variant text not null,
  scheduled_at timestamptz,
  tournament text,
  player_a_name text not null,
  player_b_name text not null,
  selected_side text not null check (selected_side in ('A','B')),
  selected_player_name text not null,
  bookmaker text not null,
  odds numeric not null check (odds > 1),
  model_probability numeric not null check (model_probability >= 0 and model_probability <= 1),
  market_probability numeric check (market_probability >= 0 and market_probability <= 1),
  edge numeric,
  ev numeric,
  fair_odds numeric,
  shadow_stake_units numeric not null default 1 check (shadow_stake_units > 0),
  closing_odds numeric check (closing_odds is null or closing_odds > 1),
  clv numeric,
  result text check (result is null or result in ('WIN','LOSS','VOID')),
  profit_units numeric,
  settled_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (provider_match_id, variant, bookmaker, selected_side)
);

create index if not exists idx_tennis_shadow_log_scheduled
  on public.tennis_shadow_log (scheduled_at desc);

create index if not exists idx_tennis_shadow_log_variant_result
  on public.tennis_shadow_log (variant, result, settled_at desc);

alter table public.tennis_shadow_log enable row level security;
revoke all on table public.tennis_shadow_log from anon, authenticated;
