import { calculateBetQuality } from "@/lib/bet-quality";

const ECONOMIC_EDGE_URL =
  "https://uciolnhvddbindxajzti.supabase.co/functions/v1/tennis-economic-runtime";

async function economicRequest(
  action: "record_bets" | "summary" | "settle",
  payload: Record<string, unknown> = {},
) {
  const cronSecret = process.env.CRON_SECRET;
  if (!cronSecret) return null;

  const response = await fetch(ECONOMIC_EDGE_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json",
    },
    cache: "no-store",
    body: JSON.stringify({
      action,
      cronSecret,
      ...payload,
    }),
  });

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      `Economic runtime ${action} failed with HTTP ${response.status}`,
    );
  }

  return body;
}

type EconomicBetRow = {
  matchId: number | string | null;
  tournament: string | null;
  scheduledTime: string | null;
  playerA: { name: string };
  playerB: { name: string };
  model: {
    mode: "full_logit" | "rank_only_fallback";
    quality: number;
  };
  market: {
    bookmaker: string;
    sourceCount?: number;
    priceSpreadRatio?: number | null;
  };
  decision: {
    side: "A" | "B";
    player: string;
    odds: number;
    modelProbability: number;
    marketProbability: number;
    edge: number;
    ev: number;
    fairOdds: number;
    sharpProbability?: number | null;
    tier: string;
    stakeUnits: number;
    bet: boolean;
  };
};

export async function recordEconomicBets(
  tour: "ATP" | "WTA",
  bets: EconomicBetRow[],
) {
  if (!bets.length) return null;

  const scored = bets
    .filter(
      (row) => row.model.mode === "full_logit" && row.decision.bet,
    )
    .map((row) => ({
      row,
      quality: calculateBetQuality({
        modelQuality: row.model.quality,
        sourceCount: row.market.sourceCount ?? 0,
        sharpProbability: row.decision.sharpProbability ?? null,
        modelProbability: row.decision.modelProbability,
        edge: row.decision.edge,
        ev: row.decision.ev,
        odds: row.decision.odds,
        priceSpreadRatio: row.market.priceSpreadRatio ?? null,
      }),
    }));

  if (!scored.length) return null;

  return economicRequest("record_bets", {
    bets: scored.map(({ row, quality }) => ({
      providerMatchId: String(row.matchId ?? ""),
      tour,
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
      tier: row.decision.tier,
      stakeUnits: row.decision.stakeUnits,
      modelMode: row.model.mode,
      modelQuality: row.model.quality,
      sourceCount: row.market.sourceCount ?? null,
      qualityScore: quality.score,
      qualityGrade: quality.grade,
      qualityComponents: quality.components,
      qualitySignals: quality.signals,
    })),
  });
}

export async function getEconomicSummary() {
  return economicRequest("summary");
}
