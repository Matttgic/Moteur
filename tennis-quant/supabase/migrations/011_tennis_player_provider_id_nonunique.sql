-- Live Tennis provider player IDs are not guaranteed to be globally unique
-- across all records/tours. The canonical application identity is
-- (tour, normalized_name), so keep provider_player_id searchable but non-unique.

alter table public.tennis_players
  drop constraint if exists tennis_players_provider_player_id_key;

create index if not exists idx_tennis_players_provider_player_id
  on public.tennis_players (provider_player_id)
  where provider_player_id is not null;
