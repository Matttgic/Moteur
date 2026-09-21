import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const BASE_URL = "https://api.livetennisapi.com/api/public/v1";
const ALLOWED_TOURS = new Set(["atp", "wta"]);

const EXCLUDED_TEAM_COMPETITIONS = [
  /davis cup/i,
  /billie jean king cup/i,
  /bjk cup/i,
  /laver cup/i,
  /hopman cup/i,
];

type ProviderPlayer = {
  id?: number;
  name?: string;
  ranking?: number | null;
  [key: string]: unknown;
};

type ProviderMatch = {
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
    p1?: ProviderPlayer;
    p2?: ProviderPlayer;
  } | null;
  [key: string]: unknown;
};

type ProviderListResponse = {
  data?: ProviderMatch[];
  meta?: unknown;
};

function isCancelled(match: ProviderMatch) {
  return (
    match.status === "cancelled" ||
    /cancelled|canceled|walkover|withdrawn|abandoned/i.test(
      match.event_status ?? ""
    )
  );
}

function isExcludedCompetition(match: ProviderMatch) {
  const tournament = match.tournament ?? "";
  return EXCLUDED_TEAM_COMPETITIONS.some((pattern) => pattern.test(tournament));
}

function modelEligibility(match: ProviderMatch) {
  const reasons: string[] = [];
  const p1 = match.players?.p1;
  const p2 = match.players?.p2;

  if (!p1?.name || !p2?.name) {
    reasons.push("missing_player_identity");
  }

  if (typeof p1?.ranking !== "number") {
    reasons.push("missing_player_1_ranking");
  }

  if (typeof p2?.ranking !== "number") {
    reasons.push("missing_player_2_ranking");
  }

  if (!match.surface) {
    reasons.push("missing_surface");
  }

  return {
    model_eligible: reasons.length === 0,
    model_ineligibility_reasons: reasons,
  };
}

export async function GET(request: NextRequest) {
  const tour = request.nextUrl.searchParams.get("tour")?.toLowerCase() ?? "atp";

  if (!ALLOWED_TOURS.has(tour)) {
    return NextResponse.json(
      { error: "unsupported_tour", allowed: ["atp", "wta"] },
      { status: 400 }
    );
  }

  const apiKey = process.env.LIVE_TENNIS_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error: "provider_not_configured",
        provider: "Live Tennis API",
        required_env: "LIVE_TENNIS_API_KEY"
      },
      { status: 503 }
    );
  }

  const url = new URL(`${BASE_URL}/matches`);
  url.searchParams.set("status", "upcoming");
  url.searchParams.set("tour", tour);
  url.searchParams.set("limit", "100");

  let response: Response;
  try {
    response = await fetch(url, {
      headers: {
        "X-API-Key": apiKey,
        "Accept": "application/json"
      },
      next: { revalidate: 300 }
    });
  } catch {
    return NextResponse.json(
      { error: "provider_unreachable", provider: "Live Tennis API" },
      { status: 502 }
    );
  }

  if (!response.ok) {
    const retryAfter = response.headers.get("retry-after");
    return NextResponse.json(
      {
        error: "provider_error",
        provider: "Live Tennis API",
        upstream_status: response.status,
        retry_after: retryAfter
      },
      { status: response.status === 429 ? 429 : 502 }
    );
  }

  const payload = (await response.json()) as ProviderListResponse;
  if (!Array.isArray(payload.data)) {
    return NextResponse.json(
      { error: "provider_contract_error", provider: "Live Tennis API" },
      { status: 502 }
    );
  }

  const unknownDrawType = payload.data.filter(
    (match) => match && typeof match.is_doubles !== "boolean"
  ).length;

  const doubles = payload.data.filter(
    (match) => match && match.is_doubles === true
  ).length;

  const cancelled = payload.data.filter(
    (match) => match && match.is_doubles === false && isCancelled(match)
  );

  const nonTourCompetitions = payload.data.filter(
    (match) =>
      match &&
      match.is_doubles === false &&
      !isCancelled(match) &&
      isExcludedCompetition(match)
  );

  const eligibleFixtures = payload.data
    .filter(
      (match) =>
        match &&
        match.is_doubles === false &&
        !isCancelled(match) &&
        !isExcludedCompetition(match)
    )
    .map((match) => ({
      ...match,
      ...modelEligibility(match),
    }));

  return NextResponse.json({
    source: "Live Tennis API",
    source_tier_required: "FREE",
    tour: tour.toUpperCase(),
    scope: "tour_singles_only",
    fetched_matches: payload.data.length,
    accepted_matches: eligibleFixtures.length,
    model_eligible_matches: eligibleFixtures.filter(
      (match) => match.model_eligible
    ).length,
    excluded_doubles: doubles,
    excluded_cancelled: cancelled.length,
    excluded_non_tour_competition: nonTourCompetitions.length,
    excluded_unknown_draw_type: unknownDrawType,
    filters: {
      singles_only: true,
      cancelled_removed: true,
      excluded_team_competitions: EXCLUDED_TEAM_COMPETITIONS.map(
        (pattern) => pattern.source
      ),
      incomplete_matches_are_never_forced_into_model: true,
    },
    data: eligibleFixtures,
    meta: payload.meta ?? null
  });
}
