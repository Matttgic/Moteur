import { NextRequest, NextResponse } from "next/server";
import {
  rankOnlyProbability,
  RANK_ONLY_BENCHMARK,
  type ModelTour,
} from "@/lib/calibrated-model";
import {
  getUpcomingTourFixtures,
  type LiveMatch,
  type LiveTour,
} from "@/lib/live-tennis";
import {
  getTennisOdds,
  type TennisTour,
} from "@/lib/oddspapi";

export const runtime = "nodejs";

const ALLOWED_TOURS = new Set<LiveTour>(["atp", "wta"]);

function normalizeName(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function looseName(value: string) {
  const parts = normalizeName(value).split(" ").filter(Boolean);
  if (!parts.length) return "";
  const last = parts.at(-1) ?? "";
  const firstInitial = parts[0]?.[0] ?? "";
  return `${last}:${firstInitial}`;
}

function pairKey(a: string, b: string, loose = false) {
  const fn = loose ? looseName : normalizeName;
  return [fn(a), fn(b)].sort().join("|");
}

function timeDistanceMs(a?: string | null, b?: string | null) {
  if (!a || !b) return Number.POSITIVE_INFINITY;
  const ta = Date.parse(a);
  const tb = Date.parse(b);
  if (!Number.isFinite(ta) || !Number.isFinite(tb)) {
    return Number.POSITIVE_INFINITY;
  }
  return Math.abs(ta - tb);
}

function findOddsFixture(
  live: LiveMatch,
  oddsFixtures: Awaited<ReturnType<typeof getTennisOdds>>["fixtures"],
) {
  const p1 = live.players?.p1?.name;
  const p2 = live.players?.p2?.name;
  if (!p1 || !p2) return null;

  const exactKey = pairKey(p1, p2);
  const looseKey = pairKey(p1, p2, true);

  const candidates = oddsFixtures
    .filter((fixture) => {
      if (!fixture.participant1Name || !fixture.participant2Name) return false;
      const exact =
        pairKey(fixture.participant1Name, fixture.participant2Name) === exactKey;
      const loose =
        pairKey(
          fixture.participant1Name,
          fixture.participant2Name,
          true,
        ) === looseKey;
      return exact || loose;
    })
    .map((fixture) => ({
      fixture,
      distance: timeDistanceMs(live.scheduled_time, fixture.startTime),
    }))
    .sort((a, b) => a.distance - b.distance);

  const best = candidates[0];
  if (!best || best.distance > 36 * 60 * 60 * 1000) return null;
  return best.fixture;
}

function alignQuote(
  live: LiveMatch,
  oddsFixture: Awaited<ReturnType<typeof getTennisOdds>>["fixtures"][number],
  bookmaker: string,
) {
  const p1 = live.players?.p1?.name;
  const p2 = live.players?.p2?.name;
  const q = oddsFixture.prices[bookmaker];

  if (!p1 || !p2 || !q) return null;
  if (!oddsFixture.participant1Name || !oddsFixture.participant2Name) return null;

  const liveP1 = normalizeName(p1);
  const liveP2 = normalizeName(p2);
  const oddsP1 = normalizeName(oddsFixture.participant1Name);
  const oddsP2 = normalizeName(oddsFixture.participant2Name);

  if (liveP1 === oddsP1 && liveP2 === oddsP2) {
    return {
      oddsA: q.odds1,
      oddsB: q.odds2,
      marketProbabilityA: q.noVigProbability1,
      marketProbabilityB: q.noVigProbability2,
      overround: q.overround,
      marketId: q.marketId,
      marketName: q.marketName,
    };
  }

  if (liveP1 === oddsP2 && liveP2 === oddsP1) {
    return {
      oddsA: q.odds2,
      oddsB: q.odds1,
      marketProbabilityA: q.noVigProbability2,
      marketProbabilityB: q.noVigProbability1,
      overround: q.overround,
      marketId: q.marketId,
      marketName: q.marketName,
    };
  }

  const looseLiveP1 = looseName(p1);
  const looseLiveP2 = looseName(p2);
  const looseOddsP1 = looseName(oddsFixture.participant1Name);
  const looseOddsP2 = looseName(oddsFixture.participant2Name);

  if (looseLiveP1 === looseOddsP1 && looseLiveP2 === looseOddsP2) {
    return {
      oddsA: q.odds1,
      oddsB: q.odds2,
      marketProbabilityA: q.noVigProbability1,
      marketProbabilityB: q.noVigProbability2,
      overround: q.overround,
      marketId: q.marketId,
      marketName: q.marketName,
    };
  }

  if (looseLiveP1 === looseOddsP2 && looseLiveP2 === looseOddsP1) {
    return {
      oddsA: q.odds2,
      oddsB: q.odds1,
      marketProbabilityA: q.noVigProbability2,
      marketProbabilityB: q.noVigProbability1,
      overround: q.overround,
      marketId: q.marketId,
      marketName: q.marketName,
    };
  }

  return null;
}

function classify(edge: number, ev: number) {
  if (edge >= 0.08 && ev >= 0.05) {
    return { tier: "PREMIUM" as const, stakeUnits: 0.5 };
  }
  if (edge >= 0.06 && ev >= 0.03) {
    return { tier: "VALUE" as const, stakeUnits: 0.25 };
  }
  if (edge >= 0.03 && ev > 0) {
    return { tier: "LEAN" as const, stakeUnits: 0 };
  }
  return { tier: "NO_BET" as const, stakeUnits: 0 };
}

export async function GET(request: NextRequest) {
  const rawTour =
    request.nextUrl.searchParams.get("tour")?.toLowerCase() ?? "atp";

  if (!ALLOWED_TOURS.has(rawTour as LiveTour)) {
    return NextResponse.json(
      { error: "unsupported_tour", allowed: ["atp", "wta"] },
      { status: 400 },
    );
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

  const bookmaker =
    request.nextUrl.searchParams.get("bookmaker")?.trim() || "winamax.fr";

  try {
    const [liveResult, oddsResult] = await Promise.all([
      getUpcomingTourFixtures(liveKey, rawTour as LiveTour),
      getTennisOdds(
        oddsKey,
        rawTour as TennisTour,
        Array.from(new Set([bookmaker, "pinnacle"])),
      ),
    ]);

    const modelTour = rawTour.toUpperCase() as ModelTour;
    const benchmark = RANK_ONLY_BENCHMARK[modelTour];

    const analyzed = [];
    const rejected = [];

    for (const live of liveResult.data) {
      const p1 = live.players?.p1;
      const p2 = live.players?.p2;

      if (!live.model_eligible || !p1?.name || !p2?.name) {
        rejected.push({
          matchId: live.id ?? null,
          playerA: p1?.name ?? null,
          playerB: p2?.name ?? null,
          reason: "live_data_incomplete",
          details: live.model_ineligibility_reasons,
        });
        continue;
      }

      if (typeof p1.ranking !== "number" || typeof p2.ranking !== "number") {
        rejected.push({
          matchId: live.id ?? null,
          playerA: p1.name,
          playerB: p2.name,
          reason: "missing_ranking",
        });
        continue;
      }

      const oddsFixture = findOddsFixture(live, oddsResult.fixtures);
      if (!oddsFixture) {
        rejected.push({
          matchId: live.id ?? null,
          playerA: p1.name,
          playerB: p2.name,
          reason: "odds_match_not_found",
        });
        continue;
      }

      const executable = alignQuote(live, oddsFixture, bookmaker);
      if (!executable) {
        rejected.push({
          matchId: live.id ?? null,
          playerA: p1.name,
          playerB: p2.name,
          reason: "bookmaker_moneyline_unavailable",
          bookmaker,
        });
        continue;
      }

      const sharp = alignQuote(live, oddsFixture, "pinnacle");
      const probabilityA = rankOnlyProbability(
        modelTour,
        p1.ranking,
        p2.ranking,
      );
      const probabilityB = 1 - probabilityA;

      const edgeA = probabilityA - executable.marketProbabilityA;
      const edgeB = probabilityB - executable.marketProbabilityB;
      const evA = probabilityA * executable.oddsA - 1;
      const evB = probabilityB * executable.oddsB - 1;

      const side = evA >= evB ? "A" : "B";
      const edge = side === "A" ? edgeA : edgeB;
      const ev = side === "A" ? evA : evB;
      const odds = side === "A" ? executable.oddsA : executable.oddsB;
      const probability = side === "A" ? probabilityA : probabilityB;
      const marketProbability =
        side === "A"
          ? executable.marketProbabilityA
          : executable.marketProbabilityB;
      const player = side === "A" ? p1.name : p2.name;
      const classification = classify(edge, ev);

      const sharpProbability = sharp
        ? side === "A"
          ? sharp.marketProbabilityA
          : sharp.marketProbabilityB
        : null;

      analyzed.push({
        matchId: live.id ?? null,
        oddsFixtureId: oddsFixture.fixtureId,
        tournament: live.tournament ?? oddsFixture.tournamentName,
        surface: live.surface ?? null,
        scheduledTime: live.scheduled_time ?? oddsFixture.startTime,
        playerA: {
          name: p1.name,
          ranking: p1.ranking,
        },
        playerB: {
          name: p2.name,
          ranking: p2.ranking,
        },
        model: {
          mode: "rank_only_fallback",
          probabilityA,
          probabilityB,
          fairOddsA: 1 / probabilityA,
          fairOddsB: 1 / probabilityB,
          trainedThrough: "2026-05-25",
          benchmark,
        },
        market: {
          bookmaker,
          oddsA: executable.oddsA,
          oddsB: executable.oddsB,
          noVigProbabilityA: executable.marketProbabilityA,
          noVigProbabilityB: executable.marketProbabilityB,
          overround: executable.overround,
          marketId: executable.marketId,
          marketName: executable.marketName,
          pinnacleNoVigProbabilityA: sharp?.marketProbabilityA ?? null,
          pinnacleNoVigProbabilityB: sharp?.marketProbabilityB ?? null,
        },
        decision: {
          side,
          player,
          odds,
          modelProbability: probability,
          marketProbability,
          edge,
          ev,
          fairOdds: 1 / probability,
          sharpProbability,
          sharpDelta:
            sharpProbability == null ? null : probability - sharpProbability,
          tier: classification.tier,
          stakeUnits: classification.stakeUnits,
          bet:
            classification.tier === "PREMIUM" ||
            classification.tier === "VALUE",
        },
      });
    }

    analyzed.sort((a, b) => {
      const tierScore = {
        PREMIUM: 4,
        VALUE: 3,
        LEAN: 2,
        NO_BET: 1,
      } as const;
      return (
        tierScore[b.decision.tier] - tierScore[a.decision.tier] ||
        b.decision.ev - a.decision.ev
      );
    });

    const bets = analyzed
      .filter((row) => row.decision.bet)
      .slice(0, 5);

    return NextResponse.json({
      generatedAt: new Date().toISOString(),
      tour: modelTour,
      status: bets.length ? "BET_OPPORTUNITIES_FOUND" : "NO_BET_TODAY",
      modelPolicy: {
        activeMode: "rank_only_fallback",
        reason:
          "The full_logit model is validated, but its dynamic Elo/form/serve-return state is not fresh enough for September 2026 on the current free history feed. Rank-only uses current rankings and is separately walk-forward validated.",
        fullModelEnabledForLive: false,
        noForcedBets: true,
      },
      bookmaker,
      sharpReference: "pinnacle",
      sourceSummary: {
        liveAccepted: liveResult.accepted_matches,
        liveModelEligible: liveResult.model_eligible_matches,
        oddsFixtures: oddsResult.fixtures.length,
        analyzed: analyzed.length,
        rejected: rejected.length,
      },
      bestBet: bets[0] ?? null,
      bets,
      allAnalyzed: analyzed,
      rejected,
    });
  } catch (error) {
    const status =
      typeof error === "object" &&
      error &&
      "status" in error &&
      typeof (error as { status?: unknown }).status === "number"
        ? (error as { status: number }).status
        : 502;

    return NextResponse.json(
      {
        error: "selection_engine_error",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: status === 429 ? 429 : 502 },
    );
  }
}
