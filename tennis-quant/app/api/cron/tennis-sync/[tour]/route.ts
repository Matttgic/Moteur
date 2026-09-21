import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 300;

const EDGE_RUNTIME_URL =
  "https://uciolnhvddbindxajzti.supabase.co/functions/v1/tennis-basic-backfill";

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

  const to = new Date();
  const from = new Date(to.getTime() - 3 * 24 * 60 * 60 * 1000);

  try {
    const edgeResponse = await fetch(EDGE_RUNTIME_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json",
      },
      cache: "no-store",
      body: JSON.stringify({
        action: "sync",
        cronSecret,
        liveApiKey: liveKey,
        tour,
        from: from.toISOString(),
        to: to.toISOString(),
      }),
    });

    const payload = await edgeResponse.json().catch(() => null);

    if (!edgeResponse.ok) {
      return NextResponse.json(
        {
          ok: false,
          tour: tour.toUpperCase(),
          error:
            payload && typeof payload === "object"
              ? payload
              : `edge_http_${edgeResponse.status}`,
        },
        { status: 500 },
      );
    }

    return NextResponse.json({
      ok: true,
      tour: tour.toUpperCase(),
      window: {
        from: from.toISOString(),
        to: to.toISOString(),
      },
      result: payload?.result ?? payload,
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
