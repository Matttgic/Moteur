import type { MatchInput, Prediction } from "./types";

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const logistic = (z: number) => 1 / (1 + Math.exp(-z));

export function removeVig(oddsA: number, oddsB: number) {
  if (oddsA <= 1 || oddsB <= 1) {
    throw new Error("Decimal odds must be greater than 1.");
  }

  const impliedA = 1 / oddsA;
  const impliedB = 1 / oddsB;
  const total = impliedA + impliedB;

  return {
    a: impliedA / total,
    b: impliedB / total,
    overround: total - 1,
  };
}

export function predictMatch(input: MatchInput): Prediction {
  const { playerA: a, playerB: b } = input;

  const eloDiff = a.elo - b.elo;
  const surfaceEloDiff = a.surfaceElo - b.surfaceElo;
  const holdBreakDiff = (a.holdPct + a.breakPct) - (b.holdPct + b.breakPct);
  const formDiff = a.formScore - b.formScore;
  const fatigueDiff = a.fatigueScore - b.fatigueScore;
  const physicalRiskDiff = a.physicalRisk - b.physicalRisk;

  // Transparent baseline, intentionally conservative.
  // These weights are placeholders until walk-forward training/backtesting replaces them.
  const z =
    0.0048 * eloDiff +
    0.0038 * surfaceEloDiff +
    0.035 * holdBreakDiff +
    0.028 * formDiff -
    0.16 * fatigueDiff -
    0.22 * physicalRiskDiff;

  const rawProbabilityA = logistic(z);

  const sparseDataPenalty = 1 - clamp(input.dataQuality, 0, 1);
  const parityPenalty = 1 - clamp(Math.abs(rawProbabilityA - 0.5) * 2, 0, 1);
  const physicalPenalty = clamp(Math.max(a.physicalRisk, b.physicalRisk), 0, 1);

  const uncertainty = clamp(
    0.50 * sparseDataPenalty +
      0.25 * parityPenalty +
      0.25 * physicalPenalty,
    0,
    0.45,
  );

  // Shrink uncertain estimates toward 50% rather than pretending false precision.
  const probabilityA = 0.5 + (rawProbabilityA - 0.5) * (1 - uncertainty);
  const probabilityB = 1 - probabilityA;

  const market = removeVig(input.oddsA, input.oddsB);
  const edgeA = probabilityA - market.a;
  const edgeB = probabilityB - market.b;
  const evA = probabilityA * input.oddsA - 1;
  const evB = probabilityB * input.oddsB - 1;

  const bestSide = evA >= evB ? "A" : "B";
  const bestEdge = bestSide === "A" ? edgeA : edgeB;
  const bestEv = bestSide === "A" ? evA : evB;

  let tier: Prediction["tier"] = "NO_BET";
  let stakeUnits = 0;

  if (bestEdge >= 0.06 && bestEv > 0.03 && uncertainty <= 0.18) {
    tier = "PREMIUM";
    stakeUnits = 0.75;
  } else if (bestEdge >= 0.04 && bestEv > 0.02 && uncertainty <= 0.24) {
    tier = "VALUE";
    stakeUnits = 0.5;
  } else if (bestEdge >= 0.02 && bestEv > 0) {
    tier = "LEAN";
    stakeUnits = 0;
  }

  const recommendation: Prediction["recommendation"] =
    tier === "PREMIUM" || tier === "VALUE"
      ? bestSide === "A"
        ? "BET_A"
        : "BET_B"
      : "NO_BET";

  const reasons = [
    `Δ Elo: ${eloDiff >= 0 ? "+" : ""}${eloDiff.toFixed(0)}`,
    `Δ Surface Elo: ${surfaceEloDiff >= 0 ? "+" : ""}${surfaceEloDiff.toFixed(0)}`,
    `Δ Hold+Break: ${holdBreakDiff >= 0 ? "+" : ""}${holdBreakDiff.toFixed(1)} pts`,
    `Δ Form: ${formDiff >= 0 ? "+" : ""}${formDiff.toFixed(1)}`,
    `Uncertainty: ${(uncertainty * 100).toFixed(1)}%`,
  ];

  return {
    playerA: a.name,
    playerB: b.name,
    rawProbabilityA,
    probabilityA,
    probabilityB,
    fairOddsA: 1 / probabilityA,
    fairOddsB: 1 / probabilityB,
    marketProbabilityA: market.a,
    marketProbabilityB: market.b,
    edgeA,
    edgeB,
    evA,
    evB,
    uncertainty,
    recommendation,
    tier,
    stakeUnits,
    reasons,
  };
}
