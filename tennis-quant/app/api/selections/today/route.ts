import { NextRequest, NextResponse } from "next/server";
import {
  FULL_MODEL_BENCHMARK,
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
import {
  loadPlayerStatesForNames,
  normalizePlayerName,
  probabilityForLiveMatch,
} from "@/lib/player-state";

export const runtime = "nodejs";

const ALLOWED_TOURS = new Set<LiveTour>(["atp", "wta"]);

function looseName(value: string) {
  const parts = normalizePlayerName(value).split(" ").filter(Boolean);
  if (!parts.length) return "";
  const last = parts.at(-1) ?? "";
  const firstInitial = parts[0]?.[0] ?? "";
  return `${last}:${firstInitial}`;
}

function pairKey(a: string, b: string, loose = false) {
  const fn = loose ? looseName : normalizePlayerName;
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
      return (
        pairKey(fixture.participant1Name, fixture.participant2Name) === exactKey ||
        pairKey(fixture.participant1Name, fixture.participant2Name, true) === looseKey
      );
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
  const quote = oddsFixture.prices[bookmaker];

  if (!p1 || !p2 || !quote) return null;
  if (!oddsFixture.participant1Name || !oddsFixture.participant2Name) return null;

  const live1 = normalizePlayerName(p1);
  const live2 = normalizePlayerName(p2);
  const odds1 = normalizePlayerName(oddsFixture.participant1Name);
  const odds2 = normalizePlayerName(oddsFixture.participant2Name);

  const direct =
    (live1 === odds1 && live2 === odds2) ||
    (looseName(p1) === looseName(oddsFixture.participant1Name) &&
      looseName(p2) === looseName(oddsFixture.participant2Name));

  const reverse =
    (live1 === odds2 && live2 === odds1) ||
    (looseName(p1) === looseName(oddsFixture.participant2Name) &&
      looseName(p2) === looseName(oddsFixture.participant1Name));

  if (!direct && !reverse) return null;

  return direct
    ? {
        oddsA: quote.odds1,
        oddsB: quote.odds2,
        marketProbabilityA: quote.noVigProbability1,
        marketProbabilityB: quote.noVigProbability2,
        overround: quote.overround,
        marketId: quote.marketId,
        marketName: quote.marketName,
      }
    : {
        oddsA: quote.odds2,
        oddsB: quote.odds1,
        marketProbabilityA: quote.noVigProbability2,
        marketProbabilityB: quote.noVigProbability1,
        overround: quote.overround,
        marketId: quote.marketId,
        marketName: quote.marketName,
      };
}

function classify(edge: number, ev: number) {
  if (edge >= 0.08 && ev >= 0.05) return { tier: "PREMIUM" as const, stakeUnits: 0.5 };
  if (edge >= 0.06 && ev >= 0.03) return { tier: "VALUE" as const, stakeUnits: 0.25 };
  if (edge >= 0.03 && ev > 0) return { tier: "LEAN" as const, stakeUnits: 0 };
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
    const playerNames = liveResult.data.flatMap((match) =>
      [match.players?.p1?.name, match.players?.p2?.name].filter(
        (value): value is string => Boolean(value),
      ),
    );
    const states = await loadPlayerStatesForNames(modelTour, playerNames);

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
      const stateA = states.get(normalizePlayerName(p1.name)) ?? null;
      const stateB = states.get(normalizePlayerName(p2.name)) ?? null;
      const model = probabilityForLiveMatch(modelTour, live, stateA, stateB);

      const edgeA = model.probabilityA - executable.marketProbabilityA;
      const edgeB = model.probabilityB - executable.marketProbabilityB;
      const evA = model.probabilityA * executable.oddsA - 1;
      const evB = model.probabilityB * executable.oddsB - 1;

      const side = evA >= evB ? "A" : "B";
      const edge = side === "A" ? edgeA : edgeB;
      const ev = side === "A" ? evA : evB;
      const odds = side === "A" ? executable.oddsA : executable.oddsB;
      const probability =
        side === "A" ? model.probabilityA : model.probabilityB;
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
        playerA: { name: p1.name, ranking: p1.ranking },
        playerB: { name: p2.name, ranking: p2.ranking },
        model: {
          mode: model.mode,
          probabilityA: model.probabilityA,
          probabilityB: model.probabilityB,
          fairOddsA: 1 / model.probabilityA,
          fairOddsB: 1 / model.probabilityB,
          quality: model.quality,
          reason: model.reason,
          features: model.features,
          trainedThrough: "2026-05-25",
          benchmark:
            model.mode === "full_logit"
              ? FULL_MODEL_BENCHMARK[modelTour]
              : RANK_ONLY_BENCHMARK[modelTour],
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
      const tierScore = { PREMIUM: 4, VALUE: 3, LEAN: 2, NO_BET: 1 } as const;
      return (
        tierScore[b.decision.tier] - tierScore[a.decision.tier] ||
        b.decision.ev - a.decision.ev
      );
    });

    const bets = analyzed.filter((row) => row.decision.bet).slice(0, 5);
    const fullModelCount = analyzed.filter(
      (row) => row.model.mode === "full_logit",
    ).length;

    return NextResponse.json({
      generatedAt: new Date().toISOString(),
      tour: modelTour,
      status: bets.length ? "BET_OPPORTUNITIES_FOUND" : "NO_BET_TODAY",
      modelPolicy: {
        fullModelEnabledForLive: true,
        fullModelMatches: fullModelCount,
        fallbackMatches: analyzed.length - fullModelCount,
        noForcedBets: true,
        fallbackPolicy:
          "Use rank_only_logit when player state is missing, stale, or has insufficient serve/return quality.",
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
