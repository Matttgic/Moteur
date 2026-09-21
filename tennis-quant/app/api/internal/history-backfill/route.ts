import { NextRequest, NextResponse } from "next/server";
import { syncCompletedHistory } from "@/lib/history-ingest";
import type { LiveTour } from "@/lib/live-tennis";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function GET(request: NextRequest) {
  if (process.env.VERCEL_ENV !== "preview") {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }

  const tour = request.nextUrl.searchParams.get("tour")?.toLowerCase();
  const from = request.nextUrl.searchParams.get("from");
  const to = request.nextUrl.searchParams.get("to");

  if ((tour !== "atp" && tour !== "wta") || !from || !to) {
    return NextResponse.json(
      { error: "invalid_params", required: "tour=atp|wta&from=YYYY-MM-DD&to=YYYY-MM-DD" },
      { status: 400 },
    );
  }

  const apiKey = process.env.LIVE_TENNIS_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "LIVE_TENNIS_API_KEY_missing" }, { status: 503 });
  }

  try {
    const result = await syncCompletedHistory({
      apiKey,
      tour: tour as LiveTour,
      from: new Date(`${from}T00:00:00.000Z`).toISOString(),
      to: new Date(`${to}T23:59:59.999Z`).toISOString(),
    });

    return NextResponse.json({
      ok: true,
      tour: tour.toUpperCase(),
      from,
      to,
      result,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        tour: tour.toUpperCase(),
        from,
        to,
        error: error instanceof Error ? error.message : "unknown_error",
      },
      { status: 500 },
    );
  }
}
