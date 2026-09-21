import { NextRequest, NextResponse } from "next/server";
import {
  recommendedSyncWindow,
  syncCompletedHistory,
} from "@/lib/history-ingest";
import type { LiveTour } from "@/lib/live-tennis";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ tour: string }> },
) {
  const cronSecret = process.env.CRON_SECRET;
  const authorization = request.headers.get("authorization");

  if (!cronSecret || authorization !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const { tour: rawTour } = await context.params;
  const tour = rawTour.toLowerCase();

  if (tour !== "atp" && tour !== "wta") {
    return NextResponse.json(
      { error: "unsupported_tour", allowed: ["atp", "wta"] },
      { status: 400 },
    );
  }

  const liveKey = process.env.LIVE_TENNIS_API_KEY;
  if (!liveKey) {
    return NextResponse.json(
      {
        error: "provider_not_configured",
        required_env: "LIVE_TENNIS_API_KEY",
      },
      { status: 503 },
    );
  }

  try {
    const window = await recommendedSyncWindow(tour as LiveTour);
    const result = await syncCompletedHistory({
      apiKey: liveKey,
      tour: tour as LiveTour,
      from: window.from,
      to: window.to,
    });

    return NextResponse.json({
      ok: true,
      tour: tour.toUpperCase(),
      window,
      result,
      schedule: request.headers.get("x-vercel-cron-schedule"),
      completedAt: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        tour: tour.toUpperCase(),
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}
