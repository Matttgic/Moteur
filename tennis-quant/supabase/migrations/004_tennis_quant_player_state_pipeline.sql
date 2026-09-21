alter table public.tennis_matches
  add column if not exists outcome text,
  add column if not exists final_score jsonb,
  add column if not exists completed_at timestamptz,
  add column if not exists provider_updated_at timestamptz;

create table if not exists public.tennis_player_states (
  player_id uuid primary key references public.tennis_players(id) on delete cascade,
  tour text not null check (tour in ('ATP','WTA')),
  elo numeric not null default 1500,
  hard_elo numeric not null default 1500,
  clay_elo numeric not null default 1500,
  grass_elo numeric not null default 1500,
  carpet_elo numeric not null default 1500,
  recent_results jsonb not null default '[]'::jsonb,
  recent_match_dates jsonb not null default '[]'::jsonb,
  service_games numeric not null default 0,
  service_holds numeric not null default 0,
  return_games numeric not null default 0,
  return_breaks numeric not null default 0,
  service_sample_matches integer not null default 0,
  data_quality numeric not null default 0,
  last_match_at timestamptz,
  state_as_of timestamptz not null default now(),
  source text not null default 'live-tennis-basic',
  feature_version text not null default 'state-v1',
  updated_at timestamptz not null default now()
);

create table if not exists public.tennis_ingestion_runs (
  id uuid primary key default gen_random_uuid(),
  job_name text not null,
  tour text check (tour in ('ATP','WTA')),
  window_from timestamptz,
  window_to timestamptz,
  status text not null check (status in ('running','success','partial','failed')),
  fetched_matches integer not null default 0,
  processed_matches integer not null default 0,
  skipped_matches integer not null default 0,
  provider_requests integer not null default 0,
  details jsonb not null default '{}'::jsonb,
  started_at timestamptz not null default now(),
  finished_at timestamptz
);

create index if not exists idx_tennis_player_states_tour_last_match
  on public.tennis_player_states (tour, last_match_at desc);

create index if not exists idx_tennis_ingestion_runs_job_started
  on public.tennis_ingestion_runs (job_name, started_at desc);

create index if not exists idx_tennis_matches_completed
  on public.tennis_matches (tour, completed_at desc)
  where completed_at is not null;

alter table public.tennis_player_states enable row level security;
alter table public.tennis_ingestion_runs enable row level security;

revoke all on table public.tennis_player_states from anon, authenticated;
revoke all on table public.tennis_ingestion_runs from anon, authenticated;
