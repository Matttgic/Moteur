revoke all on table
  public.tennis_players,
  public.tennis_matches,
  public.tennis_feature_snapshots,
  public.tennis_odds_snapshots,
  public.tennis_model_runs,
  public.tennis_predictions,
  public.tennis_outcomes
from anon, authenticated;

create index if not exists idx_tennis_features_player
  on public.tennis_feature_snapshots (player_id);

create index if not exists idx_tennis_matches_player_a
  on public.tennis_matches (player_a_id);

create index if not exists idx_tennis_matches_player_b
  on public.tennis_matches (player_b_id);

create index if not exists idx_tennis_matches_winner
  on public.tennis_matches (winner_id);

create index if not exists idx_tennis_outcomes_winner
  on public.tennis_outcomes (winner_id);

create index if not exists idx_tennis_predictions_model_run
  on public.tennis_predictions (model_run_id);

create index if not exists idx_tennis_predictions_odds_snapshot
  on public.tennis_predictions (odds_snapshot_id);
