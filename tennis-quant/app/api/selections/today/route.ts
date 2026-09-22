import { NextRequest, NextResponse } from "next/server";
import { recordEconomicBets } from "@/lib/economics";
import { recordShadowPicks } from "@/lib/ops";
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
  FRENCH_EXECUTION_BOOKMAKERS,
  getTennisOdds,
  type TennisTour,
} from "@/lib/oddspapi";
import {
  getHistorySyncFreshness,
  loadPlayerStatesForNames,
  normalizePlayerName,
  probabilityForLiveMatch,
} from "@/lib/player-state";

export const runtime = "nodejs";

const ALLOWED_TOURS = new Set<LiveTour>(["atp", "wta"]);

function canonicalExternalName(value: string) {
  const trimmed = value.trim();
  const commaIndex = trimmed.indexOf(",");

  if (commaIndex > 0) {
    const familyName = trimmed.slice(0, commaIndex).trim();
    const givenNames = trimmed.slice(commaIndex + 1).trim();

    if (familyName && givenNames) {
      return normalizePlayerName(`${givenNames} ${familyName}`);
    }
  }

  return normalizePlayerName(trimmed);
}

function looseName(value: string) {
  const parts = canonicalExternalName(value).split(" ").filter(Boolean);
  if (!parts.length) return "";
  const last = parts.at(-1) ?? "";
  const firstInitial = parts[0]?.[0] ?? "";
  return `${last}:${firstInitial}`;
}

function pairKey(a: string, b: string, loose = false) {
  const fn = loose ? looseName : canonicalExternalName;
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

  const live1 = canonicalExternalName(p1);
  const live2 = canonicalExternalName(p2);
  const odds1 = canonicalExternalName(oddsFixture.participant1Name);
  const odds2 = canonicalExternalName(oddsFixture.participant2Name);

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

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function marketGuard(input: {
  side: "A" | "B";
  executionQuotes: Array<{
    bookmaker: string;
    quote: NonNullable<ReturnType<typeof alignQuote>>;
  }>;
  odds: number;
  edge: number;
  ev: number;
  modelMode: "full_logit" | "rank_only_fallback";
  sharpProbability: number | null;
}) {
  const sideOdds = input.executionQuotes
    .map(({ quote }) => (input.side === "A" ? quote.oddsA : quote.oddsB))
    .filter((value) => Number.isFinite(value) && value > 1);

  const medianSelectedOdds = median(sideOdds);
  const priceSpreadRatio =
    medianSelectedOdds && medianSelectedOdds > 0
      ? input.odds / medianSelectedOdds
      : null;

  const reasons: string[] = [];

  if (!Number.isFinite(input.odds) || input.odds <= 1 || input.odds > 15) {
    reasons.push("odds_outlier");
  }

  if (
    sideOdds.length >= 2 &&
    priceSpreadRatio !== null &&
    priceSpreadRatio > 1.35
  ) {
    reasons.push("cross_book_price_outlier");
  }

  if (
    input.sharpProbability == null &&
    sideOdds.length < 2 &&
    (input.edge > 0.2 || input.ev > 0.5)
  ) {
    reasons.push("unconfirmed_extreme_edge");
  }

  if (
    input.modelMode === "rank_only_fallback" &&
    input.odds > 4 &&
    input.sharpProbability == null
  ) {
    reasons.push("fallback_longshot_without_sharp_confirmation");
  }

  return {
    blocked: reasons.length > 0,
    reasons,
    sourceCount: sideOdds.length,
    medianSelectedOdds,
    priceSpreadRatio,
  };
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

  const requestedBookmaker =
    request.nextUrl.searchParams.get("bookmaker")?.trim() || null;
  const executionBookmakers = requestedBookmaker
    ? [requestedBookmaker]
    : [...FRENCH_EXECUTION_BOOKMAKERS];

  try {
    const [liveResult, oddsResult] = await Promise.all([
      getUpcomingTourFixtures(liveKey, rawTour as LiveTour),
      getTennisOdds(
        oddsKey,
        rawTour as TennisTour,
        Array.from(new Set([...executionBookmakers, "pinnacle"])),
      ),
    ]);

    const modelTour = rawTour.toUpperCase() as ModelTour;
    const playerNames = liveResult.data.flatMap((match) =>
      [match.players?.p1?.name, match.players?.p2?.name].filter(
        (value): value is string => Boolean(value),
      ),
    );
    const [states, historySync] = await Promise.all([
      loadPlayerStatesForNames(modelTour, playerNames),
      getHistorySyncFreshness(modelTour),
    ]);

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

      const stateA = states.get(normalizePlayerName(p1.name)) ?? null;
      const stateB = states.get(normalizePlayerName(p2.name)) ?? null;
      const model = probabilityForLiveMatch(
        modelTour,
        live,
        stateA,
        stateB,
        historySync.fresh,
      );

      const executionQuotes: Array<{
        bookmaker: string;
        quote: NonNullable<ReturnType<typeof alignQuote>>;
      }> = [];

      for (const candidateBookmaker of executionBookmakers) {
        const quote = alignQuote(live, oddsFixture, candidateBookmaker);
        if (quote) {
          executionQuotes.push({
            bookmaker: candidateBookmaker,
            quote,
          });
        }
      }

      if (!executionQuotes.length) {
        rejected.push({
          matchId: live.id ?? null,
          playerA: p1.name,
          playerB: p2.name,
          reason: "french_bookmaker_moneyline_unavailable",
          bookmakersTried: executionBookmakers,
        });
        continue;
      }

      const executionCandidates = executionQuotes.flatMap(
        ({ bookmaker, quote }) => [
          {
            bookmaker,
            quote,
            side: "A" as const,
            ev: model.probabilityA * quote.oddsA - 1,
          },
          {
            bookmaker,
            quote,
            side: "B" as const,
            ev: model.probabilityB * quote.oddsB - 1,
          },
        ],
      );

      executionCandidates.sort((a, b) => b.ev - a.ev);
      const bestExecution = executionCandidates[0];
      const bookmaker = bestExecution.bookmaker;
      const executable = bestExecution.quote;
      const side = bestExecution.side;

      const sharp = alignQuote(live, oddsFixture, "pinnacle");
      const edgeA = model.probabilityA - executable.marketProbabilityA;
      const edgeB = model.probabilityB - executable.marketProbabilityB;
      const edge = side === "A" ? edgeA : edgeB;
      const ev = bestExecution.ev;
      const odds = side === "A" ? executable.oddsA : executable.oddsB;
      const probability =
        side === "A" ? model.probabilityA : model.probabilityB;
      const marketProbability =
        side === "A"
          ? executable.marketProbabilityA
          : executable.marketProbabilityB;
      const player = side === "A" ? p1.name : p2.name;

      const sharpProbability = sharp
        ? side === "A"
          ? sharp.marketProbabilityA
          : sharp.marketProbabilityB
        : null;

      const guard = marketGuard({
        side,
        executionQuotes,
        odds,
        edge,
        ev,
        modelMode: model.mode,
        sharpProbability,
      });

      const classification =
        guard.blocked
          ? { tier: "NO_BET" as const, stakeUnits: 0 }
          : model.mode === "rank_only_fallback"
            ? edge >= 0.03 && ev > 0
              ? { tier: "LEAN" as const, stakeUnits: 0 }
              : { tier: "NO_BET" as const, stakeUnits: 0 }
            : classify(edge, ev);

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
          sourceCount: guard.sourceCount,
          medianSelectedOdds: guard.medianSelectedOdds,
          priceSpreadRatio: guard.priceSpreadRatio,
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
            !guard.blocked &&
            model.mode === "full_logit" &&
            (classification.tier === "PREMIUM" ||
              classification.tier === "VALUE"),
          guard: {
            blocked: guard.blocked,
            reasons: guard.reasons,
          },
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

    const shadowPicks = analyzed.flatMap((row) => {
      if (
        row.model.mode !== "full_logit" ||
        row.decision.guard.blocked ||
        row.matchId == null
      ) {
        return [];
      }

      const variants: string[] = [];

      if (
        row.decision.edge >= 0.10 &&
        row.decision.ev >= 0.07 &&
        row.decision.odds <= 5
      ) {
        variants.push("strict_edge");
      }

      if (
        row.decision.edge >= 0.06 &&
        row.decision.ev >= 0.03 &&
        row.decision.odds <= 6 &&
        (row.market.sourceCount >= 2 ||
          row.decision.sharpProbability != null)
      ) {
        variants.push("market_confirmed");
      }

      if (
        row.decision.edge >= 0.05 &&
        row.decision.ev >= 0.02 &&
        row.decision.odds <= 2.5
      ) {
        variants.push("short_price");
      }

      return variants.map((variant) => ({
        variant,
        providerMatchId: String(row.matchId),
        scheduledAt: row.scheduledTime,
        tournament: row.tournament,
        playerAName: row.playerA.name,
        playerBName: row.playerB.name,
        selectedSide: row.decision.side,
        selectedPlayerName: row.decision.player,
        bookmaker: row.market.bookmaker,
        odds: row.decision.odds,
        modelProbability: row.decision.modelProbability,
        marketProbability: row.decision.marketProbability,
        edge: row.decision.edge,
        ev: row.decision.ev,
        fairOdds: row.decision.fairOdds,
      }));
    });

    let shadowTracking: {
      recorded: boolean;
      candidates: number;
      warning: string | null;
    } = {
      recorded: false,
      candidates: shadowPicks.length,
      warning: null,
    };

    try {
      const tracked = await recordShadowPicks(modelTour, shadowPicks);
      shadowTracking = {
        recorded: Boolean(tracked?.ok),
        candidates: shadowPicks.length,
        warning: null,
      };
    } catch (trackingError) {
      shadowTracking = {
        recorded: false,
        candidates: shadowPicks.length,
        warning:
          trackingError instanceof Error
            ? trackingError.message
            : "shadow_tracking_failed",
      };
    }

    let economicTracking: {
      recorded: boolean;
      warning: string | null;
    } = {
      recorded: false,
      warning: null,
    };

    try {
      const tracked = await recordEconomicBets(modelTour, bets);
      economicTracking = {
        recorded: Boolean(tracked?.ok),
        warning: null,
      };
    } catch (trackingError) {
      economicTracking = {
        recorded: false,
        warning:
          trackingError instanceof Error
            ? trackingError.message
            : "economic_tracking_failed",
      };
    }

    return NextResponse.json({
      generatedAt: new Date().toISOString(),
      tour: modelTour,
      status: bets.length ? "BET_OPPORTUNITIES_FOUND" : "NO_BET_TODAY",
      modelPolicy: {
        fullModelEnabledForLive: historySync.fresh,
        historySync,
        fullModelMatches: fullModelCount,
        fallbackMatches: analyzed.length - fullModelCount,
        noForcedBets: true,
        fallbackPolicy:
          "Rank-only fallback is informational only and can never become an actionable bet. PREMIUM/VALUE bets require full_logit.",
      },
      bookmaker: requestedBookmaker ?? "AUTO_FR",
      executionBookmakers,
      sharpReference: "pinnacle",
      economicTracking,
      shadowTracking,
      sourceSummary: {
        oddsProvider: oddsResult.provider ?? "OddsPapi",
        oddsDiscoveryMode:
          oddsResult.providerDiagnostics?.discoveryMode ?? null,
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
