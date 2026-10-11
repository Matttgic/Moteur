"use client";

import { useEffect, useMemo, useState } from "react";

type Result = "WIN" | "LOSS" | "VOID" | null;

type HistoryRow = {
  id: string;
  tour: string;
  scheduledAt: string | null;
  tournament: string | null;
  playerA: string;
  playerB: string;
  selectedPlayer: string;
  bookmaker: string;
  odds: number;
  modelProbability: number | null;
  marketProbability: number | null;
  edge: number | null;
  ev: number | null;
  tier: string;
  stakeUnits: number;
  qualityScore: number | null;
  clv: number | null;
  result: Result;
  profitUnits: number | null;
};

type Summary = {
  history?: HistoryRow[];
};

type Aggregate = {
  n: number;
  wins: number;
  losses: number;
  stake: number;
  profit: number;
  roi: number | null;
  hitRate: number | null;
  brierModel: number | null;
  brierMarket: number | null;
  logLossModel: number | null;
  logLossMarket: number | null;
};

const pct = (value: number | null) =>
  value == null ? "—" : `${(value * 100).toFixed(1)}%`;

const units = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(2)}u`;

function aggregate(rows: HistoryRow[]): Aggregate {
  const graded = rows.filter(
    (row) => row.result === "WIN" || row.result === "LOSS",
  );
  if (!graded.length) {
    return {
      n: 0,
      wins: 0,
      losses: 0,
      stake: 0,
      profit: 0,
      roi: null,
      hitRate: null,
      brierModel: null,
      brierMarket: null,
      logLossModel: null,
      logLossMarket: null,
    };
  }

  let stake = 0;
  let profit = 0;
  let wins = 0;
  let brierModel = 0;
  let brierMarket = 0;
  let logLossModel = 0;
  let logLossMarket = 0;
  let probabilityRows = 0;

  for (const row of graded) {
    stake += Number(row.stakeUnits ?? 0);
    profit += Number(row.profitUnits ?? 0);
    const y = row.result === "WIN" ? 1 : 0;
    if (y) wins += 1;

    const model = Number(row.modelProbability);
    const market = Number(row.marketProbability);
    if (
      Number.isFinite(model) &&
      Number.isFinite(market) &&
      model > 0 &&
      model < 1 &&
      market > 0 &&
      market < 1
    ) {
      probabilityRows += 1;
      brierModel += (y - model) ** 2;
      brierMarket += (y - market) ** 2;
      logLossModel += -(y * Math.log(model) + (1 - y) * Math.log(1 - model));
      logLossMarket += -(y * Math.log(market) + (1 - y) * Math.log(1 - market));
    }
  }

  return {
    n: graded.length,
    wins,
    losses: graded.length - wins,
    stake,
    profit,
    roi: stake > 0 ? profit / stake : null,
    hitRate: graded.length ? wins / graded.length : null,
    brierModel: probabilityRows ? brierModel / probabilityRows : null,
    brierMarket: probabilityRows ? brierMarket / probabilityRows : null,
    logLossModel: probabilityRows ? logLossModel / probabilityRows : null,
    logLossMarket: probabilityRows ? logLossMarket / probabilityRows : null,
  };
}

function decisionKey(row: HistoryRow) {
  const players = [row.playerA.trim().toLowerCase(), row.playerB.trim().toLowerCase()]
    .sort()
    .join("|");
  return [
    row.tour,
    row.scheduledAt ?? "",
    players,
    row.selectedPlayer.trim().toLowerCase(),
  ].join("|");
}

function dedupeBestOdds(rows: HistoryRow[]) {
  const groups = new Map<string, HistoryRow[]>();

  for (const row of rows) {
    const key = decisionKey(row);
    const current = groups.get(key) ?? [];
    current.push(row);
    groups.set(key, current);
  }

  return Array.from(groups.values()).map((group) =>
    [...group].sort((a, b) => Number(b.odds) - Number(a.odds))[0],
  );
}

function bucket(
  rows: HistoryRow[],
  specs: Array<[string, number, number | null]>,
  getter: (row: HistoryRow) => number | null,
) {
  return specs.map(([label, low, high]) => {
    const selected = rows.filter((row) => {
      const value = getter(row);
      return value != null && value >= low && (high == null || value < high);
    });
    return { label, ...aggregate(selected) };
  });
}

export default function LiveModelAudit() {
  const [data, setData] = useState<Summary | null>(null);

  useEffect(() => {
    let active = true;
    fetch("/api/economics/summary", { cache: "no-store" })
      .then((response) => response.json())
      .then((payload) => {
        if (active && payload?.summary) setData(payload.summary);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, []);

  const audit = useMemo(() => {
    const raw = data?.history ?? [];
    const graded = raw
      .filter((row) => row.result === "WIN" || row.result === "LOSS")
      .sort(
        (a, b) =>
          Date.parse(b.scheduledAt ?? "") - Date.parse(a.scheduledAt ?? ""),
      );

    const unique = dedupeBestOdds(raw);
    const overall = aggregate(unique);
    const last50 = aggregate(graded.slice(0, 50));
    const last100 = aggregate(graded.slice(0, 100));
    const value = aggregate(unique.filter((row) => row.tier === "VALUE"));
    const premium = aggregate(unique.filter((row) => row.tier === "PREMIUM"));
    const quality90 = aggregate(
      unique.filter((row) => Number(row.qualityScore) >= 90),
    );

    const calibration = bucket(
      unique,
      [
        ["<40%", 0, 0.4],
        ["40–50%", 0.4, 0.5],
        ["50–60%", 0.5, 0.6],
        ["60–70%", 0.6, 0.7],
        ["70%+", 0.7, 1.0001],
      ],
      (row) =>
        row.modelProbability == null ? null : Number(row.modelProbability),
    );

    const odds = bucket(
      unique,
      [
        ["≤1,50", 1, 1.500001],
        ["1,51–2,00", 1.500001, 2.000001],
        ["2,01–2,50", 2.000001, 2.500001],
        ["2,51–4,00", 2.500001, 4.000001],
        [">4,00", 4.000001, null],
      ],
      (row) => Number(row.odds),
    );

    const edge = bucket(
      unique,
      [
        ["<8%", -1, 0.08],
        ["8–12%", 0.08, 0.12],
        ["12–16%", 0.12, 0.16],
        ["16–20%", 0.16, 0.2],
        ["20%+", 0.2, null],
      ],
      (row) => (row.edge == null ? null : Number(row.edge)),
    );

    return {
      rawRows: raw.length,
      uniqueRows: unique.length,
      duplicateRows: Math.max(0, raw.length - unique.length),
      overall,
      last50,
      last100,
      value,
      premium,
      quality90,
      calibration,
      odds,
      edge,
    };
  }, [data]);

  if (!data) return null;

  return (
    <section className="auditPanel" aria-label="Live Model Audit">
      <div className="auditHeader">
        <div>
          <p className="eyebrow">LIVE MODEL AUDIT</p>
          <h2>Le live doit battre le marché, pas seulement le backtest</h2>
          <p>
            Rapport calculé sur les décisions uniques. Les doublons bookmakers
            sont conservés dans l&apos;historique mais retirés du bilan principal.
          </p>
        </div>
        <span className="auditBadge">
          {audit.uniqueRows} décisions · {audit.duplicateRows} lignes dupliquées
        </span>
      </div>

      <div className="auditKpis">
        <article>
          <span>ROI décisions uniques</span>
          <strong className={(audit.overall.roi ?? 0) >= 0 ? "positive" : "negative"}>
            {pct(audit.overall.roi)}
          </strong>
          <small>{units(audit.overall.profit)} · {audit.overall.n} gradées</small>
        </article>
        <article>
          <span>50 derniers</span>
          <strong className={(audit.last50.roi ?? 0) >= 0 ? "positive" : "negative"}>
            {pct(audit.last50.roi)}
          </strong>
          <small>{audit.last50.wins}-{audit.last50.losses}</small>
        </article>
        <article>
          <span>100 derniers</span>
          <strong className={(audit.last100.roi ?? 0) >= 0 ? "positive" : "negative"}>
            {pct(audit.last100.roi)}
          </strong>
          <small>{audit.last100.wins}-{audit.last100.losses}</small>
        </article>
        <article>
          <span>Brier modèle / marché</span>
          <strong>
            {audit.overall.brierModel?.toFixed(3) ?? "—"} / {audit.overall.brierMarket?.toFixed(3) ?? "—"}
          </strong>
          <small>plus bas = mieux</small>
        </article>
        <article>
          <span>Log Loss modèle / marché</span>
          <strong>
            {audit.overall.logLossModel?.toFixed(3) ?? "—"} / {audit.overall.logLossMarket?.toFixed(3) ?? "—"}
          </strong>
          <small>plus bas = mieux</small>
        </article>
        <article>
          <span>Quality ≥90</span>
          <strong className={(audit.quality90.roi ?? 0) >= 0 ? "positive" : "negative"}>
            {pct(audit.quality90.roi)}
          </strong>
          <small>{audit.quality90.n} décisions</small>
        </article>
      </div>

      <div className="auditCompare">
        <article>
          <span>VALUE · exécution</span>
          <strong className={(audit.value.roi ?? 0) >= 0 ? "positive" : "negative"}>
            ROI {pct(audit.value.roi)}
          </strong>
          <small>{audit.value.n} décisions · {audit.value.wins}-{audit.value.losses}</small>
        </article>
        <article>
          <span>PREMIUM · désormais Shadow</span>
          <strong className={(audit.premium.roi ?? 0) >= 0 ? "positive" : "negative"}>
            ROI {pct(audit.premium.roi)}
          </strong>
          <small>{audit.premium.n} décisions · suivi sans exécution</small>
        </article>
      </div>

      <div className="auditTables">
        <div>
          <strong>Calibration live</strong>
          {audit.calibration.map((row) => (
            <span key={row.label}>
              {row.label} · {row.n} · hit {pct(row.hitRate)} · ROI {pct(row.roi)}
            </span>
          ))}
        </div>
        <div>
          <strong>ROI par cote</strong>
          {audit.odds.map((row) => (
            <span key={row.label}>
              {row.label} · {row.n} · ROI {pct(row.roi)}
            </span>
          ))}
        </div>
        <div>
          <strong>ROI par edge</strong>
          {audit.edge.map((row) => (
            <span key={row.label}>
              {row.label} · {row.n} · ROI {pct(row.roi)}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
