export type Tour = "ATP" | "WTA";
export type Surface = "Hard" | "Clay" | "Grass" | "Indoor Hard";

export type PlayerSignal = {
  name: string;
  ranking?: number;
  elo: number;
  surfaceElo: number;
  holdPct: number;
  breakPct: number;
  formScore: number;
  fatigueScore: number;
  physicalRisk: number;
};

export type MatchInput = {
  tour: Tour;
  surface: Surface;
  playerA: PlayerSignal;
  playerB: PlayerSignal;
  oddsA: number;
  oddsB: number;
  dataQuality: number;
};

export type Prediction = {
  playerA: string;
  playerB: string;
  rawProbabilityA: number;
  probabilityA: number;
  probabilityB: number;
  fairOddsA: number;
  fairOddsB: number;
  marketProbabilityA: number;
  marketProbabilityB: number;
  edgeA: number;
  edgeB: number;
  evA: number;
  evB: number;
  uncertainty: number;
  recommendation: "BET_A" | "BET_B" | "NO_BET";
  tier: "PREMIUM" | "VALUE" | "LEAN" | "NO_BET";
  stakeUnits: number;
  reasons: string[];
};
