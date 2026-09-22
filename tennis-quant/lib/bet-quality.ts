export type BetQualityGrade = "A+" | "A" | "B" | "C" | "D";

export type BetQualityInput = {
  modelQuality: number;
  sourceCount: number;
  sharpProbability: number | null;
  modelProbability: number;
  edge: number;
  ev: number;
  odds: number;
  priceSpreadRatio: number | null;
};

export type BetQualityResult = {
  score: number;
  grade: BetQualityGrade;
  actionable: boolean;
  threshold: number;
  components: {
    model: number;
    marketDepth: number;
    sharpConfirmation: number;
    edge: number;
    ev: number;
    priceStability: number;
    oddsRisk: number;
  };
  signals: string[];
};

export const BET_QUALITY_ACTIONABLE_THRESHOLD = 70;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function gradeForScore(score: number): BetQualityGrade {
  if (score >= 90) return "A+";
  if (score >= 80) return "A";
  if (score >= 70) return "B";
  if (score >= 60) return "C";
  return "D";
}

function marketDepthScore(sourceCount: number) {
  if (sourceCount >= 4) return 20;
  if (sourceCount === 3) return 17;
  if (sourceCount === 2) return 13;
  if (sourceCount === 1) return 8;
  return 0;
}

function sharpScore(input: BetQualityInput) {
  if (input.sharpProbability == null) return 4;

  const delta = input.modelProbability - input.sharpProbability;
  if (delta >= 0.05) return 15;
  if (delta >= 0.02) return 13;
  if (delta >= 0) return 10;
  if (delta > -0.02) return 6;
  return 2;
}

function edgeScore(edge: number) {
  if (edge <= 0) return 0;
  if (edge < 0.03) return 4;
  if (edge < 0.06) return 8;
  if (edge < 0.10) return 12;
  if (edge <= 0.16) return 15;
  if (edge <= 0.20) return 12;
  return 7;
}

function evScore(ev: number) {
  if (ev <= 0) return 0;
  if (ev < 0.03) return 3;
  if (ev < 0.08) return 6;
  if (ev < 0.20) return 9;
  if (ev <= 0.50) return 10;
  return 6;
}

function priceStabilityScore(priceSpreadRatio: number | null) {
  if (priceSpreadRatio == null) return 4;
  if (priceSpreadRatio <= 1.05) return 10;
  if (priceSpreadRatio <= 1.10) return 9;
  if (priceSpreadRatio <= 1.20) return 6;
  if (priceSpreadRatio <= 1.35) return 3;
  return 0;
}

function oddsRiskScore(odds: number) {
  if (!Number.isFinite(odds) || odds <= 1 || odds > 15) return 0;
  if (odds <= 2.5) return 5;
  if (odds <= 4) return 4;
  if (odds <= 6) return 3;
  if (odds <= 10) return 1;
  return 0;
}

export function calculateBetQuality(input: BetQualityInput): BetQualityResult {
  const model = Math.round(clamp(input.modelQuality, 0, 1) * 25);
  const marketDepth = marketDepthScore(Math.max(0, Math.floor(input.sourceCount)));
  const sharpConfirmation = sharpScore(input);
  const edge = edgeScore(input.edge);
  const ev = evScore(input.ev);
  const priceStability = priceStabilityScore(input.priceSpreadRatio);
  const oddsRisk = oddsRiskScore(input.odds);

  const score = Math.round(
    clamp(
      model +
        marketDepth +
        sharpConfirmation +
        edge +
        ev +
        priceStability +
        oddsRisk,
      0,
      100,
    ),
  );

  const signals: string[] = [];
  const sharpDelta =
    input.sharpProbability == null
      ? null
      : input.modelProbability - input.sharpProbability;

  if (input.modelQuality >= 0.85) signals.push("strong_model_quality");
  if (input.sourceCount >= 4) signals.push("deep_market_confirmation");
  if (sharpDelta != null && sharpDelta >= 0.02) {
    signals.push("sharp_market_confirms_edge");
  }
  if (input.sharpProbability == null) signals.push("no_sharp_reference");
  if (input.priceSpreadRatio != null && input.priceSpreadRatio > 1.15) {
    signals.push("wide_cross_book_spread");
  }
  if (input.odds > 5) signals.push("longshot_price");
  if (input.edge > 0.20) signals.push("extreme_model_market_gap");
  if (score < BET_QUALITY_ACTIONABLE_THRESHOLD) {
    signals.push("below_actionable_quality_threshold");
  }

  return {
    score,
    grade: gradeForScore(score),
    actionable: score >= BET_QUALITY_ACTIONABLE_THRESHOLD,
    threshold: BET_QUALITY_ACTIONABLE_THRESHOLD,
    components: {
      model,
      marketDepth,
      sharpConfirmation,
      edge,
      ev,
      priceStability,
      oddsRisk,
    },
    signals,
  };
}
