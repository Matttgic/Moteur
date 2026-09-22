import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BASE = "https://api.oddspapi.io/v4";
const SPORT_ID = "12";

function window48h() {
  const now = new Date();
  const from = new Date(Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate(),
    0, 0, 0,
  ));
  const to = new Date(from.getTime() + 47 * 60 * 60 * 1000 + 59 * 60 * 1000);
  return { from: from.toISOString(), to: to.toISOString() };
}

async function get(path: string, params: Record<string,string>) {
  const key = process.env.ODDS_PAPI_API_KEY;
  if (!key) throw new Error("ODDS_PAPI_API_KEY_missing");
  const url = new URL(`${BASE}/${path}`);
  url.searchParams.set("apiKey", key);
  for (const [k,v] of Object.entries(params)) url.searchParams.set(k,v);
  const res = await fetch(url, { cache: "no-store", headers: { accept: "application/json" }});
  const json = await res.json().catch(() => null);
  if (!res.ok) throw new Error(`${path}_http_${res.status}`);
  return json;
}

export async function GET() {
  try {
    const window = window48h();
    const [fixturesPayload, tournamentsPayload] = await Promise.all([
      get("fixtures", {
        sportId: SPORT_ID,
        from: window.from,
        to: window.to,
        statusId: "0",
        language: "en",
      }),
      get("tournaments", {
        sportId: SPORT_ID,
        language: "en",
      }),
    ]);

    const fixtures = Array.isArray(fixturesPayload)
      ? fixturesPayload
      : Array.isArray(fixturesPayload?.data) ? fixturesPayload.data : [];
    const tournaments = Array.isArray(tournamentsPayload)
      ? tournamentsPayload
      : Array.isArray(tournamentsPayload?.data) ? tournamentsPayload.data : [];

    const unique = new Map();
    for (const f of fixtures) {
      const key = String(f?.tournamentId ?? "");
      if (!key || unique.has(key)) continue;
      unique.set(key, {
        tournamentId: f?.tournamentId ?? null,
        tournamentName: f?.tournamentName ?? null,
        tournamentSlug: f?.tournamentSlug ?? null,
        categoryName: f?.categoryName ?? null,
        categorySlug: f?.categorySlug ?? null,
        startTime: f?.startTime ?? null,
        hasOdds: f?.hasOdds ?? null,
      });
    }

    const activeTournaments = tournaments
      .filter((t:any) => Number(t?.futureFixtures ?? 0) + Number(t?.upcomingFixtures ?? 0) + Number(t?.liveFixtures ?? 0) > 0)
      .map((t:any) => ({
        tournamentId: t?.tournamentId ?? null,
        tournamentName: t?.tournamentName ?? null,
        tournamentSlug: t?.tournamentSlug ?? null,
        categoryName: t?.categoryName ?? null,
        categorySlug: t?.categorySlug ?? null,
        futureFixtures: t?.futureFixtures ?? 0,
        upcomingFixtures: t?.upcomingFixtures ?? 0,
        liveFixtures: t?.liveFixtures ?? 0,
      }));

    return NextResponse.json({
      window,
      rawFixtureCount: fixtures.length,
      fixtureTournaments: Array.from(unique.values()),
      activeTournamentCount: activeTournaments.length,
      activeTournaments,
    });
  } catch (error) {
    return NextResponse.json({
      error: error instanceof Error ? error.message : String(error),
    }, { status: 500 });
  }
}
