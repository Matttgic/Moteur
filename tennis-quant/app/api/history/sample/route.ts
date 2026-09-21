import { NextRequest, NextResponse } from "next/server";
import {
  getCompletedTourMatches,
  type LiveTour,
} from "@/lib/live-tennis";

export const runtime = "nodejs";

const ALLOWED_TOURS = new Set<LiveTour>(["atp", "wta"]);

export async function GET(request: NextRequest) {
  const tour =
    request.nextUrl.searchParams.get("tour")?.toLowerCase() ?? "atp";

  if (!ALLOWED_TOURS.has(tour as LiveTour)) {
    return NextResponse.json(
      { error: "unsupported_tour", allowed: ["atp", "wta"] },
      { status: 400 },
    );
  }

  const apiKey = process.env.LIVE_TENNIS_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "provider_not_configured" },
      { status: 503 },
    );
  }

  const to = new Date();
  const from = new Date(to.getTime() - 3 * 24 * 60 * 60 * 1000);

  try {
    const payload = await getCompletedTourMatches(
      apiKey,
      tour as LiveTour,
      from.toISOString(),
      to.toISOString(),
      3,
      0,
    );

    return NextResponse.json({
      tour: tour.toUpperCase(),
      from: from.toISOString(),
      to: to.toISOString(),
      sample: payload,
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
        error: "history_sample_error",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status },
    );
  }
}
