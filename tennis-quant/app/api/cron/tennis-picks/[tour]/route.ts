import { NextRequest, NextResponse } from "next/server";

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

  try {
    const origin = request.nextUrl.origin;
    const response = await fetch(
      `${origin}/api/selections/today?tour=${tour}`,
      {
        headers: { Accept: "application/json" },
        cache: "no-store",
      },
    );

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      return NextResponse.json(
        {
          ok: false,
          tour: tour.toUpperCase(),
          error:
            payload && typeof payload === "object"
              ? payload
              : `selection_http_${response.status}`,
        },
        { status: 500 },
      );
    }

    return NextResponse.json({
      ok: true,
      tour: tour.toUpperCase(),
      status: payload?.status ?? null,
      picksRecorded: Array.isArray(payload?.bets) ? payload.bets.length : 0,
      generatedAt: payload?.generatedAt ?? new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        tour: tour.toUpperCase(),
        error: error instanceof Error ? error.message : "unknown_error",
      },
      { status: 500 },
    );
  }
}
