import { NextRequest, NextResponse } from "next/server";
import {
  BET_QUALITY_ACTIONABLE_THRESHOLD,
  calculateBetQuality,
} from "@/lib/bet-quality";
import {
  getSelectionSnapshot,
  type SelectionTour,
} from "@/lib/selection-snapshots";

export const runtime = "nodejs";

const TOURS = new Set<SelectionTour>(["ATP", "WTA"]);

type AnalyzedRow = {
  model?: {
    mode?: string;
    quality?: number;
  };
  market?: {
    sourceCount?: number;
    priceSpreadRatio?: number | null;
  };
  decision?: {
    bet?: boolean;
    tier?: string;
    odds?: number;
    modelProbability?: number;
    edge?: number;
    ev?: number;
    sharpProbability?: number | null;
    guard?: {
      blocked?: boolean;
      reasons?: string[];
    };
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

function scoreRow(row: AnalyzedRow) {
  const model = row.model ?? {};
  const market = row.market ?? {};
  const decision = row.decision ?? {};

  const quality = calculateBetQuality({
    modelQuality: Number(model.quality ?? 0),
    sourceCount: Number(market.sourceCount ?? 0),
    sharpProbability:
      decision.sharpProbability == null
        ? null
        : Number(decision.sharpProbability),
    modelProbability: Number(decision.modelProbability ?? 0),
    edge: Number(decision.edge ?? 0),
    ev: Number(decision.ev ?? 0),
    odds: Number(decision.odds ?? 0),
    priceSpreadRatio:
      market.priceSpreadRatio == null
        ? null
        : Number(market.priceSpreadRatio),
  });

  const tierEligible =
    decision.tier === "PREMIUM" || decision.tier === "VALUE";
  const fullModel = model.mode === "full_logit";
  const guardBlocked = Boolean(decision.guard?.blocked);
  const actionable =
    fullModel && tierEligible && !guardBlocked && quality.actionable;

  return {
    ...row,
    decision: {
      ...decision,
      bet: actionable,
      quality,
      qualityGate: {
        threshold: BET_QUALITY_ACTIONABLE_THRESHOLD,
        blocked: !quality.actionable,
      },
    },
  };
}

export async function GET(request: NextRequest) {
  const tour = (
    request.nextUrl.searchParams.get("tour") ?? "atp"
  ).toUpperCase() as SelectionTour;

  if (!TOURS.has(tour)) {
    return NextResponse.json(
      { error: "unsupported_tour", allowed: ["atp", "wta"] },
      { status: 400 },
    );
  }

  try {
    const snapshot = await getSelectionSnapshot(tour);

    if (!snapshot) {
      return NextResponse.json(
        {
          error: "selection_snapshot_unavailable",
          message:
            "Aucune sélection calculée n'est encore disponible. Le prochain job privé alimentera le snapshot.",
          tour,
        },
        { status: 503 },
      );
    }

    const rawPayload = snapshot.payload as Record<string, unknown>;
    const rawAnalyzed = Array.isArray(rawPayload.allAnalyzed)
      ? (rawPayload.allAnalyzed as AnalyzedRow[])
      : [];
    const allAnalyzed = rawAnalyzed
      .map(scoreRow)
      .sort((a, b) => {
        const qa = Number(
          (a.decision as Record<string, any>)?.quality?.score ?? 0,
        );
        const qb = Number(
          (b.decision as Record<string, any>)?.quality?.score ?? 0,
        );
        const eva = Number((a.decision as Record<string, any>)?.ev ?? 0);
        const evb = Number((b.decision as Record<string, any>)?.ev ?? 0);
        return qb - qa || evb - eva;
      });

    const bets = allAnalyzed
      .filter((row) => Boolean((row.decision as Record<string, any>)?.bet))
      .slice(0, 5);
    const qualityRejected = allAnalyzed.filter((row) => {
      const decision = row.decision as Record<string, any> | undefined;
      const tierEligible =
        decision?.tier === "PREMIUM" || decision?.tier === "VALUE";
      return tierEligible && !decision?.guard?.blocked && !decision?.bet;
    }).length;

    const dataUnavailable = rawPayload.status === "DATA_UNAVAILABLE";
    const ageMinutes = Math.max(
      0,
      (Date.now() - Date.parse(snapshot.generatedAt)) / 60_000,
    );

    return NextResponse.json({
      ...rawPayload,
      status: dataUnavailable
        ? "DATA_UNAVAILABLE"
        : bets.length
          ? "BET_OPPORTUNITIES_FOUND"
          : "NO_BET_TODAY",
      bestBet: bets[0] ?? null,
      bets,
      allAnalyzed,
      qualityPolicy: {
        scoreName: "Bet Quality Score",
        scale: "0-100",
        actionableThreshold: BET_QUALITY_ACTIONABLE_THRESHOLD,
        meaning:
          "Execution robustness score, not a probability of winning or a guarantee of profit.",
        qualityRejected,
      },
      snapshot: {
        mode: "read_only",
        generatedAt: snapshot.generatedAt,
        updatedAt: snapshot.updatedAt,
        ageMinutes,
        stale: ageMinutes > 75,
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "quality_selection_read_failed",
        message: error instanceof Error ? error.message : "Unknown error",
        tour,
      },
      { status: 503 },
    );
  }
}
