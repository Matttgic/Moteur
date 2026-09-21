import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const BASE_URL = "https://api.livetennisapi.com/api/public/v1";
const ALLOWED_TOURS = new Set(["atp", "wta"]);

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
  players?: unknown;
  [key: string]: unknown;
};

type ProviderListResponse = {
  data?: ProviderMatch[];
  meta?: unknown;
};

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

  const singles = payload.data.filter(
    (match) => match && match.is_doubles === false
  );
  const unknownDrawType = payload.data.filter(
    (match) => match && typeof match.is_doubles !== "boolean"
  ).length;

  return NextResponse.json({
    source: "Live Tennis API",
    source_tier_required: "FREE",
    tour: tour.toUpperCase(),
    scope: "singles_only",
    fetched_matches: payload.data.length,
    singles_matches: singles.length,
    excluded_doubles: payload.data.length - singles.length - unknownDrawType,
    excluded_unknown_draw_type: unknownDrawType,
    data: singles,
    meta: payload.meta ?? null
  });
}
