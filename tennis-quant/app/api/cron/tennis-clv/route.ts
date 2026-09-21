import { NextRequest, NextResponse } from "next/server";
import { getTennisOdds, type TennisTour } from "@/lib/oddspapi";
import {
  getPendingClv,
  heartbeat,
  updateClvQuotes,
} from "@/lib/ops";
import { normalizePlayerName } from "@/lib/player-state";

export const runtime = "nodejs";
export const maxDuration = 300;

type PendingRow = {
  kind: "bet" | "shadow";
  id: string;
  tour: "ATP" | "WTA";
  scheduled_at: string | null;
  player_a_name: string;
  player_b_name: string;
  selected_side: "A" | "B";
  bookmaker: string;
};

function canonicalName(value: string) {
  const trimmed = value.trim();
  const comma = trimmed.indexOf(",");

  if (comma > 0) {
    const family = trimmed.slice(0, comma).trim();
    const given = trimmed.slice(comma + 1).trim();
    if (family && given) return normalizePlayerName(`${given} ${family}`);
  }

  return normalizePlayerName(trimmed);
}

function looseName(value: string) {
  const parts = canonicalName(value).split(" ").filter(Boolean);
  if (!parts.length) return "";
  return `${parts.at(-1)}:${parts[0]?.[0] ?? ""}`;
}

function pairKey(a: string, b: string, loose = false) {
  const fn = loose ? looseName : canonicalName;
  return [fn(a), fn(b)].sort().join("|");
}

function distance(a?: string | null, b?: string | null) {
  if (!a || !b) return Number.POSITIVE_INFINITY;
  const left = Date.parse(a);
  const right = Date.parse(b);
  return Number.isFinite(left) && Number.isFinite(right)
    ? Math.abs(left - right)
    : Number.POSITIVE_INFINITY;
}

export async function GET(request: NextRequest) {
  const cronSecret = process.env.CRON_SECRET;
  const authorization = request.headers.get("authorization");

  if (!cronSecret || authorization !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const oddsKey = process.env.ODDS_PAPI_API_KEY;
  if (!oddsKey) {
    return NextResponse.json(
      { error: "ODDS_PAPI_API_KEY_missing" },
      { status: 503 },
    );
  }

  try {
    const pendingPayload = await getPendingClv();
    const rows = (pendingPayload?.rows ?? []) as PendingRow[];

    if (!rows.length) {
      await heartbeat({
        jobName: "clv-refresh",
        status: "success",
        details: { pending: 0, updated: 0 },
      });

      return NextResponse.json({ ok: true, pending: 0, updated: 0 });
    }

    const updates: Array<{
      kind: "bet" | "shadow";
      id: string;
      closingOdds: number;
    }> = [];

    for (const upperTour of ["ATP", "WTA"] as const) {
      const tourRows = rows.filter((row) => row.tour === upperTour);
      if (!tourRows.length) continue;

      const bookmakers = Array.from(
        new Set(tourRows.map((row) => row.bookmaker).filter(Boolean)),
      );

      const board = await getTennisOdds(
        oddsKey,
        upperTour.toLowerCase() as TennisTour,
        bookmakers,
      );

      for (const row of tourRows) {
        const exact = pairKey(row.player_a_name, row.player_b_name);
        const loose = pairKey(row.player_a_name, row.player_b_name, true);

        const candidates = board.fixtures
          .filter((fixture) => {
            if (!fixture.participant1Name || !fixture.participant2Name) {
              return false;
            }

            return (
              pairKey(
                fixture.participant1Name,
                fixture.participant2Name,
              ) === exact ||
              pairKey(
                fixture.participant1Name,
                fixture.participant2Name,
                true,
              ) === loose
            );
          })
          .map((fixture) => ({
            fixture,
            diff: distance(row.scheduled_at, fixture.startTime),
          }))
          .sort((a, b) => a.diff - b.diff);

        const best = candidates[0];
        if (!best || best.diff > 6 * 60 * 60 * 1000) continue;

        const fixture = best.fixture;
        const quote = fixture.prices[row.bookmaker];
        if (!quote || !fixture.participant1Name || !fixture.participant2Name) {
          continue;
        }

        const a = canonicalName(row.player_a_name);
        const b = canonicalName(row.player_b_name);
        const one = canonicalName(fixture.participant1Name);
        const two = canonicalName(fixture.participant2Name);

        const direct =
          (a === one && b === two) ||
          (looseName(row.player_a_name) ===
            looseName(fixture.participant1Name) &&
            looseName(row.player_b_name) ===
              looseName(fixture.participant2Name));

        const reverse =
          (a === two && b === one) ||
          (looseName(row.player_a_name) ===
            looseName(fixture.participant2Name) &&
            looseName(row.player_b_name) ===
              looseName(fixture.participant1Name));

        if (!direct && !reverse) continue;

        const closingOdds = direct
          ? row.selected_side === "A"
            ? quote.odds1
            : quote.odds2
          : row.selected_side === "A"
            ? quote.odds2
            : quote.odds1;

        if (!Number.isFinite(closingOdds) || closingOdds <= 1) continue;

        updates.push({
          kind: row.kind,
          id: row.id,
          closingOdds,
        });
      }
    }

    const updatePayload = await updateClvQuotes(updates);

    await heartbeat({
      jobName: "clv-refresh",
      status: "success",
      details: {
        pending: rows.length,
        quotesFound: updates.length,
        updated: updatePayload?.updated ?? 0,
      },
    });

    return NextResponse.json({
      ok: true,
      pending: rows.length,
      quotesFound: updates.length,
      updated: updatePayload?.updated ?? 0,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown_error";

    try {
      await heartbeat({
        jobName: "clv-refresh",
        status: "failed",
        details: { message },
      });
    } catch {}

    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
