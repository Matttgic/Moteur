import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

const BASE = "https://api.oddspapi.io/v4";
const SPORT_ID = 12;

function window48h() {
  const now = new Date();
  const from = new Date(
    Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      0,
      0,
      0,
    ),
  );
  const to = new Date(from.getTime() + 47 * 60 * 60 * 1000 + 59 * 60 * 1000);
  return { from: from.toISOString(), to: to.toISOString() };
}

async function call(path: string, params: Record<string, string>) {
  const apiKey = process.env.ODDS_PAPI_API_KEY;
  if (!apiKey) throw new Error("ODDS_PAPI_API_KEY missing");

  const url = new URL(`${BASE}/${path}`);
  url.searchParams.set("apiKey", apiKey);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }

  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  const body = await response.json().catch(() => null);

  return {
    status: response.status,
    ok: response.ok,
    body,
  };
}

function compactOdds(body: any) {
  if (!body || typeof body !== "object") return body;

  const bookmakerOdds = body.bookmakerOdds ?? body.bookmakers ?? {};
  const books: Record<string, unknown> = {};

  for (const [slug, book] of Object.entries(bookmakerOdds as Record<string, any>)) {
    const marketKeys = Object.keys(book?.markets ?? {});
    const winner = book?.markets?.["121"] ?? null;
    books[slug] = {
      suspended: book?.suspended ?? null,
      bookmakerIsActive: book?.bookmakerIsActive ?? null,
      marketKeys: marketKeys.slice(0, 30),
      winner,
    };
  }

  return {
    fixtureId: body.fixtureId ?? null,
    participant1Name: body.participant1Name ?? null,
    participant2Name: body.participant2Name ?? null,
    tournamentId: body.tournamentId ?? null,
    tournamentName: body.tournamentName ?? null,
    statusId: body.statusId ?? null,
    hasOdds: body.hasOdds ?? null,
    bookmakerKeys: Object.keys(bookmakerOdds),
    books,
  };
}

export async function GET(request: NextRequest) {
  const tour = request.nextUrl.searchParams.get("tour")?.toLowerCase() ?? "atp";
  const { from, to } = window48h();

  const fixtures = await call("fixtures", {
    sportId: String(SPORT_ID),
    from,
    to,
    statusId: "0",
    language: "en",
  });

  const rows = Array.isArray(fixtures.body) ? fixtures.body : [];
  const target = rows.filter((fixture: any) => {
    const text = [
      fixture.tournamentName,
      fixture.tournamentSlug,
      fixture.categoryName,
      fixture.categorySlug,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    if (/challenger|itf|utr|junior|doubles?|davis cup|billie jean king|bjk cup|laver cup|hopman cup|united cup|exhibition/i.test(text)) {
      return false;
    }

    if (tour === "wta") {
      if (/\batp\b|men singles|men's singles/.test(text)) return false;
      return /\bwta\b|women singles|women's singles/.test(text);
    }

    if (/\bwta\b|women singles|women's singles/.test(text)) return false;
    return /\batp\b|men singles|men's singles/.test(text);
  });

  const samples = [];
  for (const fixture of target.slice(0, 2)) {
    if (samples.length > 0) {
      await new Promise((resolve) => setTimeout(resolve, 1100));
    }

    const odds = await call("odds", {
      fixtureId: String(fixture.fixtureId),
      bookmakers: "pinnacle,winamax.fr",
      language: "en",
      verbosity: "3",
      oddsFormat: "decimal",
    });

    samples.push({
      fixture: {
        fixtureId: fixture.fixtureId,
        participant1Name: fixture.participant1Name,
        participant2Name: fixture.participant2Name,
        tournamentId: fixture.tournamentId,
        tournamentName: fixture.tournamentName,
        categoryName: fixture.categoryName,
        statusId: fixture.statusId,
        hasOdds: fixture.hasOdds,
        startTime: fixture.startTime,
      },
      oddsStatus: odds.status,
      odds: compactOdds(odds.body),
    });
  }

  return NextResponse.json({
    tour,
    fixtureRequestStatus: fixtures.status,
    discovered: target.length,
    discoveredSample: target.slice(0, 8).map((fixture: any) => ({
      fixtureId: fixture.fixtureId,
      participant1Name: fixture.participant1Name,
      participant2Name: fixture.participant2Name,
      tournamentId: fixture.tournamentId,
      tournamentName: fixture.tournamentName,
      categoryName: fixture.categoryName,
      hasOdds: fixture.hasOdds,
      startTime: fixture.startTime,
    })),
    samples,
  });
}
