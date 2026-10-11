export const LIVE_EXECUTION_QUALITY_THRESHOLD = 90;
export const MARKET_ANCHOR_STRENGTH = 0.35;

export type CalibrationGuardResult = {
  blocked: boolean;
  reasons: string[];
};

function clampProbability(value: number) {
  return Math.min(0.995, Math.max(0.005, value));
}

function logit(value: number) {
  const p = clampProbability(value);
  return Math.log(p / (1 - p));
}

function logistic(value: number) {
  return 1 / (1 + Math.exp(-value));
}

export function calibrationGuard(
  modelProbability: number,
  edge: number,
): CalibrationGuardResult {
  const reasons: string[] = [];

  if (modelProbability >= 0.4 && modelProbability < 0.5) {
    reasons.push("live_calibration_red_zone_40_50");
  }

  if (modelProbability >= 0.6 && modelProbability < 0.7) {
    reasons.push("live_calibration_red_zone_60_70");
  }

  if (edge >= 0.2) {
    reasons.push("extreme_live_edge_red_zone");
  }

  return {
    blocked: reasons.length > 0,
    reasons,
  };
}

export function marketAnchorProbability(
  modelProbability: number,
  marketProbability: number,
  strength = MARKET_ANCHOR_STRENGTH,
) {
  const boundedStrength = Math.min(1, Math.max(0, strength));
  const marketLogit = logit(marketProbability);
  const modelLogit = logit(modelProbability);
  return clampProbability(
    logistic(marketLogit + boundedStrength * (modelLogit - marketLogit)),
  );
}

export function lineMovementPct(
  previousOdds: number | null,
  currentOdds: number,
) {
  if (
    previousOdds == null ||
    !Number.isFinite(previousOdds) ||
    previousOdds <= 1 ||
    !Number.isFinite(currentOdds) ||
    currentOdds <= 1
  ) {
    return null;
  }

  return previousOdds / currentOdds - 1;
}
