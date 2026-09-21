
alter table public.tennis_players
  add column if not exists normalized_name text;

update public.tennis_players
set normalized_name = lower(trim(regexp_replace(name, '[^[:alnum:]]+', ' ', 'g')))
where normalized_name is null;

alter table public.tennis_players
  alter column normalized_name set not null;

create unique index if not exists uq_tennis_players_tour_normalized_name
  on public.tennis_players (tour, normalized_name);
