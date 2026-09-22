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
  avgQualityScore?: number | null;
  maxDrawdownUnits: number;
  avgClv: number | null;
  clvSamples: number;
};

type Summary = {
  overall: Aggregate;
  byQuality?: Array<{ name: string } & Aggregate>;
};

const pct = (value: number | null) =>
  value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;

const units = (value: number) =>
  `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)}u`;

const gradeOrder: Record<string, number> = {
  "A+": 0,
  A: 1,
  B: 2,
  C: 3,
  D: 4,
  UNKNOWN: 5,
};

export default function QualityEconomics() {
  const [data, setData] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch("/api/economics/summary", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.error ?? "summary unavailable");
        if (active) setData(payload.summary);
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : "Erreur inconnue");
      });

    return () => {
      active = false;
    };
  }, []);

  if (error || !data) return null;

  const rows = [...(data.byQuality ?? [])].sort(
    (a, b) => (gradeOrder[a.name] ?? 99) - (gradeOrder[b.name] ?? 99),
  );

  return (
    <section className="metricGuide" aria-label="Validation du Bet Quality Score">
      <div>
        <p className="eyebrow">BET QUALITY SCORE · VALIDATION ÉCONOMIQUE</p>
        <h2>Le score doit prouver sa valeur</h2>
        <p>
          Le score ne sera considéré utile que si les grades élevés obtiennent
          progressivement de meilleurs résultats économiques et/ou une meilleure
          CLV sur un échantillon suffisant.
        </p>
        <p>
          Score moyen journalisé :{" "}
          <strong>
            {data.overall.avgQualityScore == null
              ? "collecte en cours"
              : `${Number(data.overall.avgQualityScore).toFixed(1)}/100`}
          </strong>
        </p>
      </div>

      <div>
        <strong>Performance par grade</strong>
        {rows.length ? (
          <div className="liveTags">
            {rows.map((row) => (
              <span key={row.name}>
                {row.name} · {row.settled} réglé(s) · ROI {pct(row.roi)} ·{" "}
                {units(row.profitUnits)} · CLV {pct(row.avgClv)}
              </span>
            ))}
          </div>
        ) : (
          <p>
            Les premiers paris scorés sont en cours de collecte. Les anciens paris
            restent dans l&apos;historique sans grade rétroactif inventé.
          </p>
        )}
      </div>
    </section>
  );
}
