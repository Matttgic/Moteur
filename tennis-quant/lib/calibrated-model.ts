import atpFull from "./model-specs/atp-full.json";
import wtaFull from "./model-specs/wta-full.json";
import atpRankOnly from "./model-specs/atp-rank-only.json";
import wtaRankOnly from "./model-specs/wta-rank-only.json";

export type ModelTour = "ATP" | "WTA";

export type InferenceSpec = {
  schema_version: number;
  tour: ModelTour;
  model_name: string;
  trained_through: string;
  feature_columns: string[];
  scaler: {
    mean: number[];
    scale: number[];
  };
  base_logit: {
    coefficients: number[];
    intercept: number;
  };
  platt: {
    coefficient: number;
    intercept: number;
  };
};

const FULL_SPECS: Record<ModelTour, InferenceSpec> = {
  ATP: atpFull as InferenceSpec,
  WTA: wtaFull as InferenceSpec,
};

const RANK_ONLY_SPECS: Record<ModelTour, InferenceSpec> = {
  ATP: atpRankOnly as InferenceSpec,
  WTA: wtaRankOnly as InferenceSpec,
};

export const FULL_MODEL_BENCHMARK = {
  ATP: {
    testN: 10514,
    logLoss: 0.6189421338712963,
    brier: 0.21545884548052083,
    accuracy: 0.6500856001521781,
    ece10: 0.009563552978526835,
  },
  WTA: {
    testN: 4485,
    logLoss: 0.6134117991289917,
    brier: 0.21260395436693225,
    accuracy: 0.6606465997770345,
    ece10: 0.010751161835019306,
  },
} as const;

export const RANK_ONLY_BENCHMARK = {
  ATP: {
    testN: 10514,
    logLoss: 0.6318181119281759,
    brier: 0.22102045612385035,
    accuracy: 0.6366749096442839,
    ece10: 0.008529859431814564,
  },
  WTA: {
    testN: 4485,
    logLoss: 0.6259223438968067,
    brier: 0.21787346050295286,
    accuracy: 0.652396878483835,
    ece10: 0.012503386347963287,
  },
} as const;

const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

export function calibratedProbability(
  spec: InferenceSpec,
  features: Record<string, number>,
) {
  let z = spec.base_logit.intercept;

  spec.feature_columns.forEach((feature, index) => {
    const raw = features[feature];
    if (!Number.isFinite(raw)) {
      throw new Error(`Missing or invalid feature: ${feature}`);
    }

    const scale = spec.scaler.scale[index];
    if (!Number.isFinite(scale) || scale === 0) {
      throw new Error(`Invalid scaler for feature: ${feature}`);
    }

    const standardized = (raw - spec.scaler.mean[index]) / scale;
    z += standardized * spec.base_logit.coefficients[index];
  });

  const calibratedLogit =
    spec.platt.coefficient * z + spec.platt.intercept;

  return sigmoid(calibratedLogit);
}

export function rankOnlyProbability(
  tour: ModelTour,
  rankA: number,
  rankB: number,
) {
  if (!Number.isFinite(rankA) || !Number.isFinite(rankB) || rankA < 1 || rankB < 1) {
    throw new Error("Valid positive rankings are required.");
  }

  const logRankDiff = Math.log(rankA + 1) - Math.log(rankB + 1);
  return calibratedProbability(RANK_ONLY_SPECS[tour], {
    log_rank_diff: logRankDiff,
  });
}

export function fullModelProbability(
  tour: ModelTour,
  features: Record<string, number>,
) {
  return calibratedProbability(FULL_SPECS[tour], features);
}

export function getModelSpec(tour: ModelTour, mode: "full" | "rank_only") {
  return mode === "full" ? FULL_SPECS[tour] : RANK_ONLY_SPECS[tour];
}
