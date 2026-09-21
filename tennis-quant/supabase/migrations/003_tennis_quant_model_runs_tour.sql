alter table public.tennis_model_runs
  add column if not exists tour text;

update public.tennis_model_runs
set tour = coalesce(tour, 'ATP')
where tour is null;

alter table public.tennis_model_runs
  alter column tour set not null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'tennis_model_runs_tour_check'
  ) then
    alter table public.tennis_model_runs
      add constraint tennis_model_runs_tour_check
      check (tour in ('ATP','WTA'));
  end if;
end $$;

create index if not exists idx_tennis_model_runs_tour_created
  on public.tennis_model_runs (tour, created_at desc);
