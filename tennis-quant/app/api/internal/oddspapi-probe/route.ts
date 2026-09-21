import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

const BASE = "https://api.oddspapi.io/v4";
const SPORT_ID = 12;
const WATCH = [
  "winamax.fr","winamax.es","betclic","betclic.fr","unibet","unibet.fr",
  "pmu","pmu.fr","parionssport","parionssport.fr","pinnacle",
  "bet365.fr","bet365","betfair-ex","sx.bet","sharpxch"
];

function window48h() {
  const now = new Date();
  const from = new Date(Date.UTC(
    now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), 0, 0, 0
  ));
  const to = new Date(from.getTime() + 47 * 60 * 60 * 1000 + 59 * 60 * 1000);
  return { from: from.toISOString(), to: to.toISOString() };
}

async function call(path: string, params: Record<string, string>) {
  const apiKey = process.env.ODDS_PAPI_API_KEY;
  if (!apiKey) throw new Error("ODDS_PAPI_API_KEY missing");
  const url = new URL(`${BASE}/${path}`);
  url.searchParams.set("apiKey", apiKey);
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  const body = await response.json().catch(() => null);
  return { status: response.status, body };
}

function winner(book: any) {
  const market = book?.markets?.["121"];
  const a = market?.outcomes?.["121"]?.players?.["0"]?.price;
  const b = market?.outcomes?.["122"]?.players?.["0"]?.price;
  return {
    active: book?.bookmakerIsActive ?? null,
    suspended: book?.suspended ?? null,
    winnerActive: market?.marketActive ?? null,
    odds1: typeof a === "number" ? a : null,
    odds2: typeof b === "number" ? b : null,
  };
}

export async function GET(request: NextRequest) {
  const tour = request.nextUrl.searchParams.get("tour")?.toLowerCase() ?? "atp";
  const { from, to } = window48h();

  const fixtures = await call("fixtures", {
    sportId: String(SPORT_ID), from, to, statusId: "0", language: "en",
  });

  const rows = Array.isArray(fixtures.body) ? fixtures.body : [];
  const target = rows.filter((fixture: any) => {
    const info = [
      fixture.tournamentName, fixture.tournamentSlug,
      fixture.categoryName, fixture.categorySlug,
    ].filter(Boolean).join(" ").toLowerCase();

    if (/challenger|itf|utr|junior|doubles?|davis cup|billie jean king|bjk cup|laver cup|hopman cup|united cup|exhibition/i.test(info)) return false;
    if (tour === "wta") {
      if (/\batp\b|men singles|men's singles/.test(info)) return false;
      return /\bwta\b|women singles|women's singles/.test(info);
    }
    if (/\bwta\b|women singles|women's singles/.test(info)) return false;
    return /\batp\b|men singles|men's singles/.test(info);
  });

  const samples = [];
  for (const fixture of target.slice(0, 3)) {
    if (samples.length) await new Promise((resolve) => setTimeout(resolve, 1100));
    const odds = await call("odds", {
      fixtureId: String(fixture.fixtureId),
      language: "en", verbosity: "3", oddsFormat: "decimal",
    });
    const bookmakerOdds = odds.body?.bookmakerOdds ?? odds.body?.bookmakers ?? {};
    const keys = Object.keys(bookmakerOdds);
    const watched: Record<string, unknown> = {};
    for (const slug of WATCH) watched[slug] = bookmakerOdds[slug] ? winner(bookmakerOdds[slug]) : null;

    const availableWinner = keys
      .map((slug) => ({ slug, ...winner(bookmakerOdds[slug]) }))
      .filter((row) => row.odds1 != null && row.odds2 != null)
      .slice(0, 50);

    samples.push({
      fixture: {
        fixtureId: fixture.fixtureId,
        participant1Name: fixture.participant1Name,
        participant2Name: fixture.participant2Name,
        tournamentName: fixture.tournamentName,
        startTime: fixture.startTime,
      },
      oddsStatus: odds.status,
      bookmakerCount: keys.length,
      watched,
      availableWinner,
    });
  }

  return NextResponse.json({ tour, discovered: target.length, samples });
}
