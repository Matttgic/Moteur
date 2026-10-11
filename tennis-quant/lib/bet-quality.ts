export type BetQualityGrade = "A+" | "A" | "B" | "C" | "D";

export type BetQualityInput = {
  modelQuality: number;
  sourceCount: number;
  sharpProbability: number | null;
  marketProbability: number;
  modelProbability: number;
  edge: number;
  ev: number;
  odds: number;
  priceSpreadRatio: number | null;
  lineMovementPct?: number | null;
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
    oddsBand: number;
    calibration: number;
    lineMovement: number;
  };
  signals: string[];
};

export const BET_QUALITY_ACTIONABLE_THRESHOLD = 90;

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
  if (sourceCount >= 4) return 15;
  if (sourceCount === 3) return 13;
  if (sourceCount === 2) return 10;
  if (sourceCount === 1) return 6;
  return 0;
}

function sharpScore(input: BetQualityInput) {
  if (input.sharpProbability == null) return 5;

  const sharpEdge = input.sharpProbability - input.marketProbability;
  if (sharpEdge >= 0.03) return 15;
  if (sharpEdge >= 0.01) return 13;
  if (sharpEdge >= 0) return 10;
  if (sharpEdge > -0.02) return 6;
  return 2;
}

function edgeScore(edge: number) {
  if (edge <= 0) return 0;
  if (edge < 0.03) return 3;
  if (edge < 0.06) return 6;
  if (edge < 0.08) return 10;
  if (edge < 0.10) return 8;
  if (edge < 0.12) return 6;
  if (edge < 0.16) return 4;
  if (edge < 0.20) return 2;
  return 0;
}

function evScore(ev: number) {
  if (ev <= 0) return 0;
  if (ev < 0.03) return 3;
  if (ev < 0.20) return 10;
  if (ev < 0.40) return 8;
  if (ev < 0.80) return 6;
  return 0;
}

function priceStabilityScore(priceSpreadRatio: number | null) {
  if (priceSpreadRatio == null) return 5;
  if (priceSpreadRatio <= 1.05) return 10;
  if (priceSpreadRatio <= 1.10) return 9;
  if (priceSpreadRatio <= 1.20) return 6;
  if (priceSpreadRatio <= 1.35) return 3;
  return 0;
}

function oddsBandScore(odds: number) {
  if (!Number.isFinite(odds) || odds <= 1 || odds > 15) return 0;
  if (odds <= 1.5) return 5;
  if (odds <= 2.5) return 1;
  if (odds <= 4) return 5;
  if (odds <= 6) return 2;
  return 0;
}

function calibrationScore(probability: number) {
  if (probability < 0.4) return 8;
  if (probability < 0.5) return 0;
  if (probability < 0.6) return 10;
  if (probability < 0.7) return 0;
  if (probability < 0.8) return 8;
  return 10;
}

function lineMovementScore(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return 3;
  if (value >= 0.03) return 5;
  if (value >= 0) return 4;
  if (value > -0.03) return 2;
  return 0;
}

export function calculateBetQuality(input: BetQualityInput): BetQualityResult {
  const model = Math.round(clamp(input.modelQuality, 0, 1) * 20);
  const marketDepth = marketDepthScore(Math.max(0, Math.floor(input.sourceCount)));
  const sharpConfirmation = sharpScore(input);
  const edge = edgeScore(input.edge);
  const ev = evScore(input.ev);
  const priceStability = priceStabilityScore(input.priceSpreadRatio);
  const oddsBand = oddsBandScore(input.odds);
  const calibration = calibrationScore(input.modelProbability);
  const lineMovement = lineMovementScore(input.lineMovementPct);

  const score = Math.round(
    clamp(
      model +
        marketDepth +
        sharpConfirmation +
        edge +
        ev +
        priceStability +
        oddsBand +
        calibration +
        lineMovement,
      0,
      100,
    ),
  );

  const signals: string[] = ["quality_v2_live_evidence"];
  const sharpDelta =
    input.sharpProbability == null
      ? null
      : input.sharpProbability - input.marketProbability;

  if (input.modelQuality >= 0.85) signals.push("strong_model_quality");
  if (input.sourceCount >= 4) signals.push("deep_market_confirmation");
  if (sharpDelta != null && sharpDelta >= 0.01) {
    signals.push("sharp_market_confirms_edge");
  }
  if (input.sharpProbability == null) signals.push("no_sharp_reference");
  if (input.priceSpreadRatio != null && input.priceSpreadRatio > 1.15) {
    signals.push("wide_cross_book_spread");
  }
  if (input.odds > 5) signals.push("longshot_price");
  if (input.odds > 1.5 && input.odds <= 2.5) {
    signals.push("live_odds_band_underperforming");
  }
  if (
    (input.modelProbability >= 0.4 && input.modelProbability < 0.5) ||
    (input.modelProbability >= 0.6 && input.modelProbability < 0.7)
  ) {
    signals.push("live_calibration_red_zone");
  }
  if (input.edge >= 0.20) signals.push("extreme_model_market_gap");
  if (input.lineMovementPct != null && input.lineMovementPct >= 0.03) {
    signals.push("market_moved_with_pick");
  }
  if (input.lineMovementPct != null && input.lineMovementPct <= -0.03) {
    signals.push("market_moved_against_pick");
  }
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
      oddsBand,
      calibration,
      lineMovement,
    },
    signals,
  };
}
