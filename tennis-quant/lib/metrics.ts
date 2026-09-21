export type SettledPrediction = {
  probability: number;
  won: boolean;
  odds: number;
  stake: number;
  closingOdds?: number;
};

const safeProbability = (p: number) =>
  Math.min(1 - 1e-12, Math.max(1e-12, p));

export function brierScore(rows: SettledPrediction[]) {
  if (!rows.length) return null;
  return (
    rows.reduce((sum, row) => {
      const y = row.won ? 1 : 0;
      return sum + (row.probability - y) ** 2;
    }, 0) / rows.length
  );
}

export function logLoss(rows: SettledPrediction[]) {
  if (!rows.length) return null;
  return (
    -rows.reduce((sum, row) => {
      const p = safeProbability(row.probability);
      return sum + (row.won ? Math.log(p) : Math.log(1 - p));
    }, 0) / rows.length
  );
}

export function bettingSummary(rows: SettledPrediction[]) {
  const staked = rows.reduce((sum, row) => sum + row.stake, 0);
  const profit = rows.reduce((sum, row) => {
    return sum + (row.won ? row.stake * (row.odds - 1) : -row.stake);
  }, 0);

  const clvRows = rows.filter((row) => row.closingOdds && row.closingOdds > 1);
  const avgClv =
    clvRows.length === 0
      ? null
      : clvRows.reduce((sum, row) => {
          return sum + row.odds / (row.closingOdds as number) - 1;
        }, 0) / clvRows.length;

  return {
    bets: rows.length,
    staked,
    profit,
    roi: staked > 0 ? profit / staked : null,
    hitRate:
      rows.length > 0
        ? rows.filter((row) => row.won).length / rows.length
        : null,
    averageClv: avgClv,
  };
}
