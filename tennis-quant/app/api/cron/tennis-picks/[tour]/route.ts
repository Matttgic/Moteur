import { NextRequest, NextResponse } from "next/server";
import { isCronAuthorized } from "@/lib/cron-auth";\nimport { heartbeat } from "@/lib/ops";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ tour: string }> },
) {
  if (!isCronAuthorized(request)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const cronSecret = process.env.CRON_SECRET;
  if (!cronSecret) {
    return NextResponse.json({ error: "CRON_SECRET_missing" }, { status: 503 });
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
      `${origin}/api/selections/today?tour=${tour}&refresh=1`,
      {
        headers: {
          Accept: "application/json",
          Authorization: `Bearer ${cronSecret}`,
        },
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

    const picksRecorded = Array.isArray(payload?.bets) ? payload.bets.length : 0;

    await heartbeat({
      jobName: "pick-capture",
      tour: tour.toUpperCase() as "ATP" | "WTA",
      status: "success",
      details: {
        status: payload?.status ?? null,
        picksRecorded,
        generatedAt: payload?.generatedAt ?? null,
      },
    });

    return NextResponse.json({
      ok: true,
      tour: tour.toUpperCase(),
      status: payload?.status ?? null,
      picksRecorded,
      generatedAt: payload?.generatedAt ?? new Date().toISOString(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown_error";

    try {
      await heartbeat({
        jobName: "pick-capture",
        tour: tour.toUpperCase() as "ATP" | "WTA",
        status: "failed",
        details: { message },
      });
    } catch {}

    return NextResponse.json(
      {
        ok: false,
        tour: tour.toUpperCase(),
        error: message,
      },
      { status: 500 },
    );
  }
}
