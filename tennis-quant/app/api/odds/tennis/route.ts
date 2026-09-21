import { NextRequest, NextResponse } from "next/server";
import {
  FRENCH_EXECUTION_BOOKMAKERS,
  getTennisOdds,
  type TennisTour,
} from "@/lib/oddspapi";

export const runtime = "nodejs";

const ALLOWED_TOURS = new Set<TennisTour>(["atp", "wta"]);

export async function GET(request: NextRequest) {
  const rawTour = request.nextUrl.searchParams.get("tour")?.toLowerCase() ?? "atp";
  if (!ALLOWED_TOURS.has(rawTour as TennisTour)) {
    return NextResponse.json(
      { error: "unsupported_tour", allowed: ["atp", "wta"] },
      { status: 400 },
    );
  }

  const apiKey = process.env.ODDS_PAPI_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error: "provider_not_configured",
        provider: "OddsPapi",
        required_env: "ODDS_PAPI_API_KEY",
      },
      { status: 503 },
    );
  }

  const requestedBookmaker =
    request.nextUrl.searchParams.get("bookmaker")?.trim() || null;

  const bookmakers = requestedBookmaker
    ? Array.from(new Set([requestedBookmaker, "pinnacle"]))
    : Array.from(
        new Set([...FRENCH_EXECUTION_BOOKMAKERS, "pinnacle"]),
      );

  try {
    const result = await getTennisOdds(
      apiKey,
      rawTour as TennisTour,
      bookmakers,
    );

    return NextResponse.json({
      source: "OddsPapi",
      sport: "tennis",
      scope: "ATP_WTA_main_tour_singles",
      price_policy: {
        execution_mode: requestedBookmaker ? "requested_bookmaker" : "auto_fr",
        user_bookmaker: requestedBookmaker ?? "auto_fr",
        french_bookmakers: requestedBookmaker
          ? [requestedBookmaker]
          : FRENCH_EXECUTION_BOOKMAKERS,
        sharp_reference: "pinnacle",
        odds_format: "decimal",
        vig_removed: true,
        cache_hours: 12,
      },
      ...result,
    });
  } catch (error) {
    const status =
      typeof error === "object" &&
      error &&
      "status" in error &&
      typeof (error as { status?: unknown }).status === "number"
        ? (error as { status: number }).status
        : 502;

    return NextResponse.json(
      {
        error: "odds_provider_error",
        provider: "OddsPapi",
        message: error instanceof Error ? error.message : "Unknown OddsPapi error",
      },
      { status: status === 429 ? 429 : 502 },
    );
  }
}
