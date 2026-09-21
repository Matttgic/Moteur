import { NextRequest, NextResponse } from "next/server";
import {
  getUpcomingTourFixtures,
  type LiveTour,
} from "@/lib/live-tennis";

export const runtime = "nodejs";

const ALLOWED_TOURS = new Set<LiveTour>(["atp", "wta"]);

export async function GET(request: NextRequest) {
  const rawTour =
    request.nextUrl.searchParams.get("tour")?.toLowerCase() ?? "atp";

  if (!ALLOWED_TOURS.has(rawTour as LiveTour)) {
    return NextResponse.json(
      { error: "unsupported_tour", allowed: ["atp", "wta"] },
      { status: 400 },
    );
  }

  const apiKey = process.env.LIVE_TENNIS_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error: "provider_not_configured",
        provider: "Live Tennis API",
        required_env: "LIVE_TENNIS_API_KEY",
      },
      { status: 503 },
    );
  }

  try {
    return NextResponse.json(
      await getUpcomingTourFixtures(apiKey, rawTour as LiveTour),
    );
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
        error: "provider_error",
        provider: "Live Tennis API",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: status === 429 ? 429 : 502 },
    );
  }
}
