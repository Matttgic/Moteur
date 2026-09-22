create table if not exists public.tennis_selection_snapshots (
  tour text primary key check (tour in ('ATP', 'WTA')),
  payload jsonb not null,
  generated_at timestamptz not null,
  updated_at timestamptz not null default now()
);

alter table public.tennis_selection_snapshots enable row level security;

revoke all on table public.tennis_selection_snapshots from anon, authenticated;

comment on table public.tennis_selection_snapshots is
  'Server-only latest ATP/WTA selection snapshots. Public clients read them only through trusted Next.js routes.';
