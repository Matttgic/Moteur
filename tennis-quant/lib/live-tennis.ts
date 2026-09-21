const BASE_URL = "https://api.livetennisapi.com/api/public/v1";

export type LiveTour = "atp" | "wta";

export type LivePlayer = {
  id?: number;
  name?: string;
  ranking?: number | null;
  birthday?: string | null;
  country?: string | null;
  hand?: string | null;
  [key: string]: unknown;
};

export type LiveMatch = {
  id?: number;
  tournament?: string;
  surface?: "hard" | "clay" | "grass" | null;
  indoor?: boolean;
  format?: "BO3" | "BO5" | null;
  round?: string | null;
  status?: "upcoming" | "live" | "completed" | "cancelled";
  event_status?: string | null;
  is_doubles?: boolean;
  scheduled_time?: string | null;
  players?: {
    p1?: LivePlayer;
    p2?: LivePlayer;
  } | null;
  [key: string]: unknown;
};

type ProviderListResponse = {
  data?: LiveMatch[];
  meta?: unknown;
};

const EXCLUDED_TEAM_COMPETITIONS = [
  /davis cup/i,
  /billie jean king cup/i,
  /bjk cup/i,
  /laver cup/i,
  /hopman cup/i,
  /united cup/i,
];

function isCancelled(match: LiveMatch) {
  return (
    match.status === "cancelled" ||
    /cancelled|canceled|walkover|withdrawn|abandoned/i.test(
      match.event_status ?? "",
    )
  );
}

function isExcludedCompetition(match: LiveMatch) {
  const tournament = match.tournament ?? "";
  return EXCLUDED_TEAM_COMPETITIONS.some((pattern) => pattern.test(tournament));
}

function modelEligibility(match: LiveMatch) {
  const reasons: string[] = [];
  const p1 = match.players?.p1;
  const p2 = match.players?.p2;

  if (!p1?.name || !p2?.name) reasons.push("missing_player_identity");
  if (typeof p1?.ranking !== "number") reasons.push("missing_player_1_ranking");
  if (typeof p2?.ranking !== "number") reasons.push("missing_player_2_ranking");
  if (!match.surface) reasons.push("missing_surface");

  return {
    model_eligible: reasons.length === 0,
    model_ineligibility_reasons: reasons,
  };
}

export async function getUpcomingTourFixtures(
  apiKey: string,
  tour: LiveTour,
) {
  const url = new URL(`${BASE_URL}/matches`);
  url.searchParams.set("status", "upcoming");
  url.searchParams.set("tour", tour);
  url.searchParams.set("limit", "100");

  const response = await fetch(url, {
    headers: {
      "X-API-Key": apiKey,
      Accept: "application/json",
    },
    next: { revalidate: 300 },
  });

  const payload = (await response.json().catch(() => null)) as ProviderListResponse | null;

  if (!response.ok) {
    const error = new Error(
      `Live Tennis API failed with HTTP ${response.status}`,
    );
    Object.assign(error, {
      status: response.status,
      payload,
      retryAfter: response.headers.get("retry-after"),
    });
    throw error;
  }

  if (!payload || !Array.isArray(payload.data)) {
    throw new Error("Live Tennis API contract error.");
  }

  const unknownDrawType = payload.data.filter(
    (match) => match && typeof match.is_doubles !== "boolean",
  ).length;

  const doubles = payload.data.filter(
    (match) => match && match.is_doubles === true,
  ).length;

  const cancelled = payload.data.filter(
    (match) => match && match.is_doubles === false && isCancelled(match),
  ).length;

  const nonTourCompetitions = payload.data.filter(
    (match) =>
      match &&
      match.is_doubles === false &&
      !isCancelled(match) &&
      isExcludedCompetition(match),
  ).length;

  const data = payload.data
    .filter(
      (match) =>
        match &&
        match.is_doubles === false &&
        !isCancelled(match) &&
        !isExcludedCompetition(match),
    )
    .map((match) => ({
      ...match,
      ...modelEligibility(match),
    }));

  return {
    source: "Live Tennis API",
    source_tier_required: "FREE",
    tour: tour.toUpperCase(),
    scope: "tour_singles_only",
    fetched_matches: payload.data.length,
    accepted_matches: data.length,
    model_eligible_matches: data.filter((match) => match.model_eligible).length,
    excluded_doubles: doubles,
    excluded_cancelled: cancelled,
    excluded_non_tour_competition: nonTourCompetitions,
    excluded_unknown_draw_type: unknownDrawType,
    filters: {
      singles_only: true,
      cancelled_removed: true,
      excluded_team_competitions: EXCLUDED_TEAM_COMPETITIONS.map(
        (pattern) => pattern.source,
      ),
      incomplete_matches_are_never_forced_into_model: true,
    },
    data,
    meta: payload.meta ?? null,
  };
}


type HistoryCoverageResponse = {
  data?: unknown;
  meta?: unknown;
  [key: string]: unknown;
};

export async function getHistoryCoverage(apiKey: string) {
  const response = await fetch(`${BASE_URL}/history/coverage`, {
    headers: {
      "X-API-Key": apiKey,
      Accept: "application/json",
    },
    next: { revalidate: 3600 },
  });

  const payload = (await response.json().catch(() => null)) as HistoryCoverageResponse | null;

  if (!response.ok) {
    const error = new Error(
      `Live Tennis API history coverage failed with HTTP ${response.status}`,
    );
    Object.assign(error, {
      status: response.status,
      payload,
    });
    throw error;
  }

  if (!payload) {
    throw new Error("Live Tennis API history coverage contract error.");
  }

  return payload;
}

export async function getCompletedTourMatches(
  apiKey: string,
  tour: LiveTour,
  from: string,
  to: string,
  limit = 100,
  offset = 0,
) {
  const url = new URL(`${BASE_URL}/history/matches`);
  url.searchParams.set("tour", tour);
  url.searchParams.set("draw", "singles");
  url.searchParams.set("from", from);
  url.searchParams.set("to", to);
  url.searchParams.set("limit", String(Math.min(Math.max(limit, 1), 100)));
  url.searchParams.set("offset", String(Math.max(offset, 0)));

  const response = await fetch(url, {
    headers: {
      "X-API-Key": apiKey,
      Accept: "application/json",
    },
    next: { revalidate: 300 },
  });

  const payload = (await response.json().catch(() => null)) as ProviderListResponse | null;

  if (!response.ok) {
    const error = new Error(
      `Live Tennis API completed history failed with HTTP ${response.status}`,
    );
    Object.assign(error, {
      status: response.status,
      payload,
      retryAfter: response.headers.get("retry-after"),
    });
    throw error;
  }

  if (!payload || !Array.isArray(payload.data)) {
    throw new Error("Live Tennis API completed history contract error.");
  }

  return payload;
}
