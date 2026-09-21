import { fullModelProbability, rankOnlyProbability, type ModelTour } from "./calibrated-model";
import { getSupabaseServerClient } from "./supabase";
import type { LiveMatch, LivePlayer } from "./live-tennis";

export type SurfaceKey = "hard" | "clay" | "grass" | "carpet";

export type PlayerStateSnapshot = {
  playerId: string;
  name: string;
  normalizedName: string;
  tour: ModelTour;
  elo: number;
  hardElo: number;
  clayElo: number;
  grassElo: number;
  carpetElo: number;
  recentResults: number[];
  recentMatchDates: string[];
  serviceGames: number;
  serviceHolds: number;
  returnGames: number;
  returnBreaks: number;
  dataQuality: number;
  lastMatchAt: string | null;
  stateAsOf: string;
  source: string;
};

export type ModelDecisionInput = {
  mode: "full_logit" | "rank_only_fallback";
  probabilityA: number;
  probabilityB: number;
  features: Record<string, number> | null;
  quality: number;
  reason: string;
};

const numeric = (value: unknown, fallback = 0) => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asStringArray = (value: unknown) =>
  Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];

const asNumberArray = (value: unknown) =>
  Array.isArray(value)
    ? value
        .map((item) => Number(item))
        .filter((item) => Number.isFinite(item))
        .map((item) => (item > 0 ? 1 : 0))
    : [];

export function normalizePlayerName(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function expectedScore(ratingA: number, ratingB: number) {
  return 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
}

export function surfaceKey(value?: string | null): SurfaceKey {
  const normalized = (value ?? "").toLowerCase();
  if (normalized === "clay") return "clay";
  if (normalized === "grass") return "grass";
  if (normalized === "carpet") return "carpet";
  return "hard";
}

export function surfaceElo(state: PlayerStateSnapshot, surface?: string | null) {
  switch (surfaceKey(surface)) {
    case "clay":
      return state.clayElo;
    case "grass":
      return state.grassElo;
    case "carpet":
      return state.carpetElo;
    default:
      return state.hardElo;
  }
}

export function smoothedRate(
  successes: number,
  trials: number,
  prior: number,
  strength = 24,
) {
  return (successes + prior * strength) / (trials + strength);
}

export function form10(state: PlayerStateSnapshot) {
  if (!state.recentResults.length) return 0.5;
  const wins = state.recentResults.reduce((sum, result) => sum + result, 0);
  return (wins + 2) / (state.recentResults.length + 4);
}

export function load14(state: PlayerStateSnapshot, at: Date) {
  const cutoff = at.getTime() - 14 * 24 * 60 * 60 * 1000;
  return state.recentMatchDates.filter((value) => {
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) && timestamp >= cutoff;
  }).length;
}

function ageYears(player: LivePlayer | undefined, at: Date) {
  if (!player?.birthday) return 27;
  const born = Date.parse(player.birthday);
  if (!Number.isFinite(born)) return 27;
  return Math.max(15, Math.min(50, (at.getTime() - born) / (365.2425 * 24 * 60 * 60 * 1000)));
}

function stateFreshEnough(state: PlayerStateSnapshot, at: Date) {
  if (!state.lastMatchAt) return false;
  const last = Date.parse(state.lastMatchAt);
  if (!Number.isFinite(last)) return false;
  const inactiveDays = (at.getTime() - last) / (24 * 60 * 60 * 1000);
  return inactiveDays <= 180;
}

export function fullModelEligibility(
  stateA: PlayerStateSnapshot | null,
  stateB: PlayerStateSnapshot | null,
  at: Date,
) {
  if (!stateA || !stateB) {
    return { eligible: false, quality: 0, reason: "missing_player_state" };
  }

  const quality = Math.min(stateA.dataQuality, stateB.dataQuality);

  if (!stateFreshEnough(stateA, at) || !stateFreshEnough(stateB, at)) {
    return { eligible: false, quality, reason: "stale_player_state" };
  }

  if (
    stateA.serviceGames < 24 ||
    stateB.serviceGames < 24 ||
    stateA.returnGames < 24 ||
    stateB.returnGames < 24
  ) {
    return { eligible: false, quality, reason: "insufficient_serve_return_sample" };
  }

  if (quality < 0.65) {
    return { eligible: false, quality, reason: "state_quality_below_threshold" };
  }

  return { eligible: true, quality, reason: "fresh_full_state" };
}

export function buildFullFeatures(
  tour: ModelTour,
  match: LiveMatch,
  stateA: PlayerStateSnapshot,
  stateB: PlayerStateSnapshot,
) {
  const p1 = match.players?.p1;
  const p2 = match.players?.p2;
  const rankA = typeof p1?.ranking === "number" ? p1.ranking : 1000;
  const rankB = typeof p2?.ranking === "number" ? p2.ranking : 1000;
  const at = new Date(match.scheduled_time ?? Date.now());
  const holdPrior = tour === "ATP" ? 0.8 : 0.72;
  const breakPrior = 1 - holdPrior;
  const surface = surfaceKey(match.surface);

  return {
    elo_diff: stateA.elo - stateB.elo,
    surface_elo_diff: surfaceElo(stateA, surface) - surfaceElo(stateB, surface),
    log_rank_diff: Math.log(rankA + 1) - Math.log(rankB + 1),
    age_diff: ageYears(p1, at) - ageYears(p2, at),
    hold_diff:
      smoothedRate(stateA.serviceHolds, stateA.serviceGames, holdPrior) -
      smoothedRate(stateB.serviceHolds, stateB.serviceGames, holdPrior),
    break_diff:
      smoothedRate(stateA.returnBreaks, stateA.returnGames, breakPrior) -
      smoothedRate(stateB.returnBreaks, stateB.returnGames, breakPrior),
    form10_diff: form10(stateA) - form10(stateB),
    load14_diff: load14(stateA, at) - load14(stateB, at),
    surface_hard: surface === "hard" ? 1 : 0,
    surface_clay: surface === "clay" ? 1 : 0,
    surface_grass: surface === "grass" ? 1 : 0,
  };
}

export function probabilityForLiveMatch(
  tour: ModelTour,
  match: LiveMatch,
  stateA: PlayerStateSnapshot | null,
  stateB: PlayerStateSnapshot | null,
): ModelDecisionInput {
  const p1 = match.players?.p1;
  const p2 = match.players?.p2;
  const at = new Date(match.scheduled_time ?? Date.now());
  const eligibility = fullModelEligibility(stateA, stateB, at);

  if (
    eligibility.eligible &&
    stateA &&
    stateB &&
    typeof p1?.ranking === "number" &&
    typeof p2?.ranking === "number"
  ) {
    const features = buildFullFeatures(tour, match, stateA, stateB);
    const probabilityA = fullModelProbability(tour, features);
    return {
      mode: "full_logit",
      probabilityA,
      probabilityB: 1 - probabilityA,
      features,
      quality: eligibility.quality,
      reason: eligibility.reason,
    };
  }

  if (typeof p1?.ranking !== "number" || typeof p2?.ranking !== "number") {
    throw new Error("Current rankings are required for the fallback model.");
  }

  const probabilityA = rankOnlyProbability(tour, p1.ranking, p2.ranking);
  return {
    mode: "rank_only_fallback",
    probabilityA,
    probabilityB: 1 - probabilityA,
    features: null,
    quality: eligibility.quality,
    reason: eligibility.reason,
  };
}

export async function loadPlayerStatesForNames(
  tour: ModelTour,
  names: string[],
) {
  const supabase = getSupabaseServerClient() as any;
  if (!supabase) return new Map<string, PlayerStateSnapshot>();

  const normalizedNames = Array.from(
    new Set(names.map(normalizePlayerName).filter(Boolean)),
  );
  if (!normalizedNames.length) return new Map<string, PlayerStateSnapshot>();

  const { data: players, error: playerError } = await supabase
    .from("tennis_players")
    .select("id,name,normalized_name,tour")
    .eq("tour", tour)
    .in("normalized_name", normalizedNames);

  if (playerError) throw playerError;
  if (!players?.length) return new Map<string, PlayerStateSnapshot>();

  const ids = players.map((player: any) => player.id);
  const { data: states, error: stateError } = await supabase
    .from("tennis_player_states")
    .select("*")
    .in("player_id", ids);

  if (stateError) throw stateError;

  const playerById = new Map<string, { id: string; name: string; normalized_name: string; tour: string }>(
    players.map((player: any) => [player.id, player]),
  );
  const result = new Map<string, PlayerStateSnapshot>();

  for (const row of states ?? []) {
    const player = playerById.get(row.player_id);
    if (!player) continue;

    result.set(player.normalized_name, {
      playerId: row.player_id,
      name: player.name,
      normalizedName: player.normalized_name,
      tour,
      elo: numeric(row.elo, 1500),
      hardElo: numeric(row.hard_elo, 1500),
      clayElo: numeric(row.clay_elo, 1500),
      grassElo: numeric(row.grass_elo, 1500),
      carpetElo: numeric(row.carpet_elo, 1500),
      recentResults: asNumberArray(row.recent_results).slice(-10),
      recentMatchDates: asStringArray(row.recent_match_dates).slice(-30),
      serviceGames: numeric(row.service_games),
      serviceHolds: numeric(row.service_holds),
      returnGames: numeric(row.return_games),
      returnBreaks: numeric(row.return_breaks),
      dataQuality: numeric(row.data_quality),
      lastMatchAt: row.last_match_at ?? null,
      stateAsOf: row.state_as_of,
      source: row.source,
    });
  }

  return result;
}

export function defaultPlayerState(
  playerId: string,
  name: string,
  tour: ModelTour,
): PlayerStateSnapshot {
  return {
    playerId,
    name,
    normalizedName: normalizePlayerName(name),
    tour,
    elo: 1500,
    hardElo: 1500,
    clayElo: 1500,
    grassElo: 1500,
    carpetElo: 1500,
    recentResults: [],
    recentMatchDates: [],
    serviceGames: 0,
    serviceHolds: 0,
    returnGames: 0,
    returnBreaks: 0,
    dataQuality: 0.35,
    lastMatchAt: null,
    stateAsOf: new Date(0).toISOString(),
    source: "live-tennis-basic",
  };
}

function setSurfaceElo(
  state: PlayerStateSnapshot,
  surface: SurfaceKey,
  value: number,
) {
  if (surface === "clay") state.clayElo = value;
  else if (surface === "grass") state.grassElo = value;
  else if (surface === "carpet") state.carpetElo = value;
  else state.hardElo = value;
}

export function applyCompletedMatchStateUpdate(
  winner: PlayerStateSnapshot,
  loser: PlayerStateSnapshot,
  surfaceValue: string | null | undefined,
  playedAt: string,
) {
  const surface = surfaceKey(surfaceValue);

  const globalExpected = expectedScore(winner.elo, loser.elo);
  winner.elo += 28 * (1 - globalExpected);
  loser.elo += 28 * (0 - globalExpected);

  const winnerSurface = surfaceElo(winner, surface);
  const loserSurface = surfaceElo(loser, surface);
  const surfaceExpected = expectedScore(winnerSurface, loserSurface);
  setSurfaceElo(winner, surface, winnerSurface + 32 * (1 - surfaceExpected));
  setSurfaceElo(loser, surface, loserSurface + 32 * (0 - surfaceExpected));

  winner.recentResults = [...winner.recentResults, 1].slice(-10);
  loser.recentResults = [...loser.recentResults, 0].slice(-10);
  winner.recentMatchDates = [...winner.recentMatchDates, playedAt].slice(-30);
  loser.recentMatchDates = [...loser.recentMatchDates, playedAt].slice(-30);

  winner.lastMatchAt = playedAt;
  loser.lastMatchAt = playedAt;
  winner.stateAsOf = playedAt;
  loser.stateAsOf = playedAt;
  winner.source = "live-tennis-basic";
  loser.source = "live-tennis-basic";

  const serviceQuality = (state: PlayerStateSnapshot) => {
    const sample = Math.min(state.serviceGames, state.returnGames);
    if (sample >= 120) return 0.95;
    if (sample >= 60) return 0.85;
    if (sample >= 24) return 0.7;
    return 0.4;
  };

  winner.dataQuality = Math.max(winner.dataQuality, serviceQuality(winner));
  loser.dataQuality = Math.max(loser.dataQuality, serviceQuality(loser));
}

export function stateToDatabaseRow(state: PlayerStateSnapshot) {
  return {
    player_id: state.playerId,
    tour: state.tour,
    elo: state.elo,
    hard_elo: state.hardElo,
    clay_elo: state.clayElo,
    grass_elo: state.grassElo,
    carpet_elo: state.carpetElo,
    recent_results: state.recentResults,
    recent_match_dates: state.recentMatchDates,
    service_games: state.serviceGames,
    service_holds: state.serviceHolds,
    return_games: state.returnGames,
    return_breaks: state.returnBreaks,
    service_sample_matches: 0,
    data_quality: state.dataQuality,
    last_match_at: state.lastMatchAt,
    state_as_of: state.stateAsOf,
    source: state.source,
    feature_version: "state-v1",
    updated_at: new Date().toISOString(),
  };
}
