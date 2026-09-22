alter table public.tennis_bet_log
  add column if not exists quality_score numeric,
  add column if not exists quality_grade text,
  add column if not exists quality_components jsonb,
  add column if not exists quality_signals jsonb;

alter table public.tennis_bet_log
  drop constraint if exists tennis_bet_log_quality_score_check;

alter table public.tennis_bet_log
  add constraint tennis_bet_log_quality_score_check
  check (quality_score is null or (quality_score >= 0 and quality_score <= 100));

create index if not exists idx_tennis_bet_log_quality_grade
  on public.tennis_bet_log (quality_grade, scheduled_at desc);
