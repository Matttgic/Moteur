"use client";

import { useEffect, useState } from "react";

type Aggregate = {
  total: number;
  settled: number;
  pending: number;
  wins: number;
  losses: number;
  stakeUnits: number;
  profitUnits: number;
  roi: number | null;
  hitRate: number | null;
  avgOdds: number | null;
  avgEdge: number | null;
  avgEv: number | null;
  maxDrawdownUnits: number;
  avgClv: number | null;
  clvSamples: number;
};

type EconomicSummary = {
  generatedAt: string;
  sampleStatus: "COLLECTING" | "EARLY_SAMPLE" | "VALIDATING" | "ROBUST_SAMPLE";
  overall: Aggregate;
  byTour: Array<{ name: string } & Aggregate>;
  byModel: Array<{ name: string } & Aggregate>;
  byTier: Array<{ name: string } & Aggregate>;
  last10: Array<{
    result: "WIN" | "LOSS";
    tour: string;
    player: string;
    bookmaker: string;
    odds: number;
    stakeUnits: number;
    profitUnits: number;
    modelMode: string;
    tier: string;
    scheduledAt: string;
  }>;
};

const pct = (value: number | null) =>
  value == null ? "—" : `${(value * 100).toFixed(1)}%`;

const units = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(2)}u`;

function statusText(status: EconomicSummary["sampleStatus"]) {
  if (status === "ROBUST_SAMPLE") return "ÉCHANTILLON ROBUSTE";
  if (status === "VALIDATING") return "VALIDATION EN COURS";
  if (status === "EARLY_SAMPLE") return "ÉCHANTILLON PRÉCOCE";
  return "COLLECTE EN COURS";
}

export default function EconomicValidation() {
  const [data, setData] = useState<EconomicSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    fetch("/api/economics/summary", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload?.error ?? "Economic validation unavailable");
        }
        if (active) setData(payload.summary);
      })
      .catch((err) => {
        if (active) {
          setError(err instanceof Error ? err.message : "Erreur inconnue");
        }
      });

    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return (
      <article className="economicGate">
        <div>
          <p className="eyebrow">ECONOMIC VALIDATION</p>
          <h3>Collecte temporairement indisponible</h3>
        </div>
        <p>
          Le moteur de paris reste actif. Seul le tableau économique n'a pas pu
          charger ses statistiques.
        </p>
        <span>{error}</span>
      </article>
    );
  }

  if (!data) {
    return (
      <article className="economicGate">
        <div>
          <p className="eyebrow">ECONOMIC VALIDATION</p>
          <h3>Chargement du journal réel…</h3>
        </div>
        <p>
          Les picks proposés sont enregistrés puis comparés aux résultats
          officiels pour mesurer la performance économique réelle.
        </p>
        <span>ROI · Drawdown · CLV</span>
      </article>
    );
  }

  const stats = data.overall;

  return (
    <section className="economicPanel" aria-label="Validation économique">
      <div className="economicHeader">
        <div>
          <p className="eyebrow">ECONOMIC VALIDATION · LIVE TRACKING</p>
          <h2>Performance des picks réellement proposés</h2>
          <p>
            Pas de ROI théorique : uniquement les sélections réellement sorties
            par le moteur, avec leurs cotes au moment de la décision.
          </p>
        </div>
        <span className="economicStatus">{statusText(data.sampleStatus)}</span>
      </div>

      <div className="economicStats">
        <article>
          <span>Paris enregistrés</span>
          <strong>{stats.total}</strong>
          <small>{stats.pending} en attente</small>
        </article>
        <article>
          <span>Bilan réglé</span>
          <strong>{stats.wins}-{stats.losses}</strong>
          <small>{stats.settled} paris réglés</small>
        </article>
        <article>
          <span>Profit net</span>
          <strong className={stats.profitUnits >= 0 ? "positive" : "negative"}>
            {units(stats.profitUnits)}
          </strong>
          <small>{stats.stakeUnits.toFixed(2)}u engagées</small>
        </article>
        <article>
          <span>ROI réel</span>
          <strong className={(stats.roi ?? 0) >= 0 ? "positive" : "negative"}>
            {pct(stats.roi)}
          </strong>
          <small>profit / mises</small>
        </article>
        <article>
          <span>Hit rate</span>
          <strong>{pct(stats.hitRate)}</strong>
          <small>paris gagnants</small>
        </article>
        <article>
          <span>Cote moyenne</span>
          <strong>{stats.avgOdds?.toFixed(2) ?? "—"}</strong>
          <small>au moment du pick</small>
        </article>
        <article>
          <span>Max drawdown</span>
          <strong>{stats.maxDrawdownUnits.toFixed(2)}u</strong>
          <small>baisse depuis un pic</small>
        </article>
        <article>
          <span>CLV moyen</span>
          <strong>{pct(stats.avgClv)}</strong>
          <small>{stats.clvSamples} cote(s) de clôture</small>
        </article>
      </div>

      {data.last10.length ? (
        <div className="economicRecent">
          <strong>10 derniers paris réglés</strong>
          <div>
            {data.last10.map((row, index) => (
              <span
                className={row.result === "WIN" ? "recentWin" : "recentLoss"}
                key={`${row.player}-${row.scheduledAt}-${index}`}
              >
                {row.result === "WIN" ? "W" : "L"} · {row.player} @{Number(row.odds).toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      ) : (
        <p className="economicCollecting">
          Les premiers picks sont maintenant enregistrés. Le ROI restera marqué
          comme précoce tant que l'échantillon de paris réglés est insuffisant.
        </p>
      )}
    </section>
  );
}
