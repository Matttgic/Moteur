import { createHash, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { getUpcomingTourFixtures } from "@/lib/live-tennis";
import { getTennisOdds } from "@/lib/oddspapi";
import { heartbeat } from "@/lib/ops";

export const runtime = "nodejs";
export const maxDuration = 300;

const SCHEDULER_TOKEN_SHA256 =
  "56e58d13e49669747926615b2cff1617d571549a94f1632789414aa63d1f4c75";

function isAuthorized(request: NextRequest) {
  const authorization = request.headers.get("authorization");
  const cronSecret = process.env.CRON_SECRET;

  if (cronSecret && authorization === `Bearer ${cronSecret}`) {
    return true;
  }

  if (!authorization?.startsWith("Bearer ")) {
    return false;
  }

  const token = authorization.slice("Bearer ".length);
  const actual = createHash("sha256").update(token).digest();
  const expected = Buffer.from(SCHEDULER_TOKEN_SHA256, "hex");

  return (
    actual.length === expected.length &&
    timingSafeEqual(actual, expected)
  );
}

export async function GET(request: NextRequest) {
  if (!isAuthorized(request)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const liveKey = process.env.LIVE_TENNIS_API_KEY;
  const oddsKey = process.env.ODDS_PAPI_API_KEY;

  if (!liveKey || !oddsKey) {
    return NextResponse.json(
      {
        error: "provider_not_configured",
        missing: [
          !liveKey ? "LIVE_TENNIS_API_KEY" : null,
          !oddsKey ? "ODDS_PAPI_API_KEY" : null,
        ].filter(Boolean),
      },
      { status: 503 },
    );
  }

  const results: Record<string, unknown> = {};

  for (const tour of ["atp", "wta"] as const) {
    const upper = tour.toUpperCase() as "ATP" | "WTA";

    try {
      const live = await getUpcomingTourFixtures(liveKey, tour);
      results[`live_${tour}`] = {
        ok: true,
        accepted: live.accepted_matches,
        eligible: live.model_eligible_matches,
      };

      await heartbeat({
        jobName: "provider-health-live",
        tour: upper,
        status: "success",
        details: results[`live_${tour}`] as Record<string, unknown>,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown_error";
      results[`live_${tour}`] = { ok: false, message };

      await heartbeat({
        jobName: "provider-health-live",
        tour: upper,
        status: "failed",
        details: { message },
      });
    }

    try {
      const odds = await getTennisOdds(oddsKey, tour, ["bet365.fr"]);
      results[`odds_${tour}`] = {
        ok: true,
        discoveredFixtures: odds.discoveredFixtures ?? 0,
        pricedFixtures: odds.fixtures.length,
      };

      await heartbeat({
        jobName: "provider-health-odds",
        tour: upper,
        status: "success",
        details: results[`odds_${tour}`] as Record<string, unknown>,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown_error";
      results[`odds_${tour}`] = { ok: false, message };

      await heartbeat({
        jobName: "provider-health-odds",
        tour: upper,
        status: "failed",
        details: { message },
      });
    }
  }

  return NextResponse.json({
    ok: true,
    checkedAt: new Date().toISOString(),
    results,
  });
}
