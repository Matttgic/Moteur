import type { MatchInput } from "./types";

export const demoMatches: MatchInput[] = [
  {
    tour: "ATP",
    surface: "Hard",
    playerA: {
      name: "Player A",
      ranking: 8,
      elo: 2015,
      surfaceElo: 2040,
      holdPct: 86.2,
      breakPct: 24.4,
      formScore: 7.8,
      fatigueScore: 1.5,
      physicalRisk: 0.05
    },
    playerB: {
      name: "Player B",
      ranking: 19,
      elo: 1940,
      surfaceElo: 1925,
      holdPct: 82.8,
      breakPct: 21.7,
      formScore: 6.4,
      fatigueScore: 2.5,
      physicalRisk: 0.08
    },
    oddsA: 1.64,
    oddsB: 2.26,
    dataQuality: 0.92
  },
  {
    tour: "WTA",
    surface: "Clay",
    playerA: {
      name: "Player C",
      ranking: 31,
      elo: 1840,
      surfaceElo: 1905,
      holdPct: 69.8,
      breakPct: 39.6,
      formScore: 7.1,
      fatigueScore: 1.2,
      physicalRisk: 0.04
    },
    playerB: {
      name: "Player D",
      ranking: 22,
      elo: 1880,
      surfaceElo: 1845,
      holdPct: 72.1,
      breakPct: 36.8,
      formScore: 6.6,
      fatigueScore: 2.1,
      physicalRisk: 0.06
    },
    oddsA: 2.04,
    oddsB: 1.79,
    dataQuality: 0.89
  }
];
