import {
  getCompletedTourMatches,
  isExcludedCompetition,
  type LiveMatch,
  type LivePlayer,
  type LiveTour,
} from "./live-tennis";
import {
  applyCompletedMatchStateUpdate,
  defaultPlayerState,
  normalizePlayerName,
  stateToDatabaseRow,
  type PlayerStateSnapshot,
} from "./player-state";
import { getSupabaseServerClient } from "./supabase";
import type { ModelTour } from "./calibrated-model";

const BATCH_SIZE = 150;
const MAX_HISTORY_PAGES = 100;
export const STATE_SEED_AS_OF = "2026-05-25";

function chunk<T>(values: T[], size = BATCH_SIZE) {
  const output: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    output.push(values.slice(index, index + size));
  }
  return output;
}

function modelTour(tour: LiveTour): ModelTour {
  return tour.toUpperCase() as ModelTour;
}

function playedAt(match: LiveMatch) {
  const candidates = [
    match.live_at,
    match.event_status_updated_at,
    match.updated_at,
    match.scheduled_time,
  ];

  for (const candidate of candidates) {
    if (!candidate) continue;
    const timestamp = Date.parse(candidate);
    if (Number.isFinite(timestamp)) return new Date(timestamp).toISOString();
  }

  return null;
}

function validPlayer(player: LivePlayer | undefined) {
  return Boolean(player?.name?.trim());
}

function eligibleCompletedMatch(match: LiveMatch) {
  return (
    typeof match.id === "number" &&
    match.outcome === "completed" &&
    (match.winner === 1 || match.winner === 2) &&
    validPlayer(match.players?.p1) &&
    validPlayer(match.players?.p2) &&
    Boolean(match.surface) &&
    !isExcludedCompetition(match) &&
    Boolean(playedAt(match))
  );
}

async function fetchCompletedWindow(
  apiKey: string,
  tour: LiveTour,
  from: string,
  to: string,
) {
  const matches: LiveMatch[] = [];
  let offset = 0;
  let requests = 0;

  for (let page = 0; page < MAX_HISTORY_PAGES; page += 1) {
    const payload = await getCompletedTourMatches(
      apiKey,
      tour,
      from.slice(0, 10),
      to.slice(0, 10),
      100,
      offset,
    );

    requests += 1;
    matches.push(...(payload.data ?? []));

    const meta = payload.meta ?? {};
    if (meta.has_more === false) break;

    const limit =
      typeof meta.limit === "number" && meta.limit > 0 ? meta.limit : 100;

    if (meta.has_more !== true && (payload.data?.length ?? 0) < limit) {
      break;
    }

    offset += limit;
  }

  return { matches, requests };
}

async function existingMatchIds(supabase: any, ids: string[]) {
  const existing = new Set<string>();

  for (const batch of chunk(ids)) {
    const { data, error } = await supabase
      .from("tennis_matches")
      .select("provider_match_id")
      .in("provider_match_id", batch);

    if (error) throw error;

    for (const row of data ?? []) {
      if (row.provider_match_id) existing.add(row.provider_match_id);
    }
  }

  return existing;
}

function playerRow(tour: ModelTour, player: LivePlayer) {
  const name = player.name?.trim() ?? "";
  return {
    provider_player_id:
      typeof player.id === "number" ? String(player.id) : null,
    tour,
    name,
    normalized_name: normalizePlayerName(name),
    country_code: player.country ?? null,
    handedness: player.hand ?? null,
    birth_date: player.birthday ?? null,
    updated_at: new Date().toISOString(),
  };
}

async function ensurePlayers(
  supabase: any,
  tour: ModelTour,
  matches: LiveMatch[],
) {
  const rows = new Map<string, ReturnType<typeof playerRow>>();

  for (const match of matches) {
    for (const player of [match.players?.p1, match.players?.p2]) {
      if (!player?.name) continue;
      const row = playerRow(tour, player);
      rows.set(row.normalized_name, row);
    }
  }

  for (const batch of chunk(Array.from(rows.values()))) {
    const { error } = await supabase
      .from("tennis_players")
      .upsert(batch, { onConflict: "tour,normalized_name" });

    if (error) throw error;
  }

  const players: any[] = [];
  const names = Array.from(rows.keys());

  for (const batch of chunk(names)) {
    const { data, error } = await supabase
      .from("tennis_players")
      .select("id,name,normalized_name,provider_player_id,tour")
      .eq("tour", tour)
      .in("normalized_name", batch);

    if (error) throw error;
    players.push(...(data ?? []));
  }

  return players;
}

async function loadStates(
  supabase: any,
  tour: ModelTour,
  players: any[],
) {
  const states = new Map<string, PlayerStateSnapshot>();
  const playerById = new Map(players.map((player) => [player.id, player]));

  for (const batch of chunk(players.map((player) => player.id))) {
    const { data, error } = await supabase
      .from("tennis_player_states")
      .select("*")
      .in("player_id", batch);

    if (error) throw error;

    for (const row of data ?? []) {
      const player = playerById.get(row.player_id);
      if (!player) continue;

      states.set(player.normalized_name, {
        playerId: row.player_id,
        name: player.name,
        normalizedName: player.normalized_name,
        tour,
        elo: Number(row.elo),
        hardElo: Number(row.hard_elo),
        clayElo: Number(row.clay_elo),
        grassElo: Number(row.grass_elo),
        carpetElo: Number(row.carpet_elo),
        recentResults: Array.isArray(row.recent_results)
          ? row.recent_results.map(Number).filter(Number.isFinite).slice(-10)
          : [],
        recentMatchDates: Array.isArray(row.recent_match_dates)
          ? row.recent_match_dates.filter(
              (value: unknown): value is string => typeof value === "string",
            ).slice(-30)
          : [],
        serviceGames: Number(row.service_games),
        serviceHolds: Number(row.service_holds),
        returnGames: Number(row.return_games),
        returnBreaks: Number(row.return_breaks),
        dataQuality: Number(row.data_quality),
        lastMatchAt: row.last_match_at ?? null,
        stateAsOf: row.state_as_of,
        source: row.source,
      });
    }
  }

  for (const player of players) {
    if (!states.has(player.normalized_name)) {
      states.set(
        player.normalized_name,
        defaultPlayerState(player.id, player.name, tour),
      );
    }
  }

  return states;
}

export async function syncCompletedHistory(input: {
  apiKey: string;
  tour: LiveTour;
  from: string;
  to: string;
}) {
  const supabase = getSupabaseServerClient() as any;
  if (!supabase) {
    throw new Error("Supabase server client is not configured.");
  }

  const tour = modelTour(input.tour);
  const startedAt = new Date().toISOString();

  const { data: run, error: runError } = await supabase
    .from("tennis_ingestion_runs")
    .insert({
      job_name: "completed-history-sync",
      tour,
      window_from: input.from,
      window_to: input.to,
      status: "running",
      started_at: startedAt,
    })
    .select("id")
    .single();

  if (runError) throw runError;

  try {
    const fetched = await fetchCompletedWindow(
      input.apiKey,
      input.tour,
      input.from,
      input.to,
    );

    const eligible = fetched.matches
      .filter(eligibleCompletedMatch)
      .sort((a, b) => {
        const left = Date.parse(playedAt(a) ?? "1970-01-01T00:00:00Z");
        const right = Date.parse(playedAt(b) ?? "1970-01-01T00:00:00Z");
        return left - right || (a.id ?? 0) - (b.id ?? 0);
      });

    const providerIds = eligible.map((match) => String(match.id));
    const existing = await existingMatchIds(supabase, providerIds);
    const pending = eligible.filter(
      (match) => !existing.has(String(match.id)),
    );

    const players = await ensurePlayers(supabase, tour, pending);
    const playerMap = new Map(
      players.map((player) => [player.normalized_name, player]),
    );
    const states = await loadStates(supabase, tour, players);

    const matchRows: any[] = [];
    const outcomeDrafts: Array<{
      providerMatchId: string;
      winnerId: string;
      playedAt: string;
      score: unknown;
    }> = [];

    let skipped = fetched.matches.length - eligible.length;

    for (const match of pending) {
      const p1 = match.players?.p1;
      const p2 = match.players?.p2;
      const when = playedAt(match);

      if (!p1?.name || !p2?.name || !when) {
        skipped += 1;
        continue;
      }

      const key1 = normalizePlayerName(p1.name);
      const key2 = normalizePlayerName(p2.name);
      const player1 = playerMap.get(key1);
      const player2 = playerMap.get(key2);
      const state1 = states.get(key1);
      const state2 = states.get(key2);

      if (!player1 || !player2 || !state1 || !state2) {
        skipped += 1;
        continue;
      }

      const winnerState = match.winner === 1 ? state1 : state2;
      const loserState = match.winner === 1 ? state2 : state1;
      const winnerId = match.winner === 1 ? player1.id : player2.id;

      applyCompletedMatchStateUpdate(
        winnerState,
        loserState,
        match.surface,
        when,
      );

      const providerMatchId = String(match.id);
      matchRows.push({
        provider_match_id: providerMatchId,
        tour,
        tournament: match.tournament ?? "Unknown",
        round: match.round ?? null,
        surface: match.surface ?? "hard",
        indoor: Boolean(match.indoor),
        scheduled_at: match.scheduled_time ?? when,
        player_a_id: player1.id,
        player_b_id: player2.id,
        winner_id: winnerId,
        status: "completed",
        outcome: match.outcome,
        final_score: match.score ?? null,
        completed_at: when,
        provider_updated_at: match.updated_at ?? match.event_status_updated_at ?? null,
        updated_at: new Date().toISOString(),
      });

      outcomeDrafts.push({
        providerMatchId,
        winnerId,
        playedAt: when,
        score: match.score ?? null,
      });
    }

    const matchIdByProvider = new Map<string, string>();

    for (const batch of chunk(matchRows)) {
      const { data, error } = await supabase
        .from("tennis_matches")
        .upsert(batch, { onConflict: "provider_match_id" })
        .select("id,provider_match_id");

      if (error) throw error;

      for (const row of data ?? []) {
        if (row.provider_match_id) {
          matchIdByProvider.set(row.provider_match_id, row.id);
        }
      }
    }

    const outcomes = outcomeDrafts
      .map((draft) => {
        const matchId = matchIdByProvider.get(draft.providerMatchId);
        if (!matchId) return null;

        return {
          match_id: matchId,
          settled_at: draft.playedAt,
          winner_id: draft.winnerId,
          retired: false,
          walkover: false,
          score:
            draft.score == null
              ? null
              : typeof draft.score === "string"
                ? draft.score
                : JSON.stringify(draft.score),
        };
      })
      .filter(Boolean);

    for (const batch of chunk(outcomes as any[])) {
      const { error } = await supabase
        .from("tennis_outcomes")
        .upsert(batch, { onConflict: "match_id" });

      if (error) throw error;
    }

    const stateRows = Array.from(states.values()).map(stateToDatabaseRow);
    for (const batch of chunk(stateRows)) {
      const { error } = await supabase
        .from("tennis_player_states")
        .upsert(batch, { onConflict: "player_id" });

      if (error) throw error;
    }

    const summary = {
      fetched: fetched.matches.length,
      eligible: eligible.length,
      alreadyProcessed: eligible.length - pending.length,
      processed: matchRows.length,
      skipped,
      providerRequests: fetched.requests,
    };

    const { error: finishError } = await supabase
      .from("tennis_ingestion_runs")
      .update({
        status: "success",
        fetched_matches: summary.fetched,
        processed_matches: summary.processed,
        skipped_matches: summary.skipped,
        provider_requests: summary.providerRequests,
        details: summary,
        finished_at: new Date().toISOString(),
      })
      .eq("id", run.id);

    if (finishError) throw finishError;

    return summary;
  } catch (error) {
    await supabase
      .from("tennis_ingestion_runs")
      .update({
        status: "failed",
        details: {
          message: error instanceof Error ? error.message : "Unknown error",
        },
        finished_at: new Date().toISOString(),
      })
      .eq("id", run.id);

    throw error;
  }
}

export async function recommendedSyncWindow(tour: LiveTour) {
  const supabase = getSupabaseServerClient() as any;
  if (!supabase) {
    throw new Error("Supabase server client is not configured.");
  }

  const upperTour = modelTour(tour);
  const { data, error } = await supabase
    .from("tennis_matches")
    .select("completed_at")
    .eq("tour", upperTour)
    .not("completed_at", "is", null)
    .order("completed_at", { ascending: false })
    .limit(1);

  if (error) throw error;

  const latest = data?.[0]?.completed_at
    ? new Date(data[0].completed_at)
    : new Date(`${STATE_SEED_AS_OF}T00:00:00.000Z`);

  const from = new Date(latest.getTime() - 3 * 24 * 60 * 60 * 1000);
  const seedFloor = new Date("2026-05-26T00:00:00.000Z");
  if (from < seedFloor) from.setTime(seedFloor.getTime());

  return {
    from: from.toISOString(),
    to: new Date().toISOString(),
  };
}
