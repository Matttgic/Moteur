"use client";

import { useEffect, useState } from "react";

type Variant = {
  variant: string;
  total: number;
  settled: number;
  pending: number;
  wins: number;
  losses: number;
  profitUnits: number;
  roi: number | null;
  hitRate: number | null;
  avgOdds: number | null;
  avgClv: number | null;
  clvSamples: number;
  maxDrawdownUnits: number;
  reviewEligible: boolean;
};

const labels: Record<string, string> = {
  strict_edge: "Edge strict",
  market_confirmed: "Marché confirmé",
  short_price: "Cotes courtes",
};

const pct = (value: number | null) =>
  value == null ? "—" : `${(value * 100).toFixed(1)}%`;

export default function ShadowLab() {
  const [variants, setVariants] = useState<Variant[]>([]);

  useEffect(() => {
    let active = true;

    fetch("/api/shadow/summary", { cache: "no-store" })
      .then((response) => response.json())
      .then((payload) => {
        if (active && Array.isArray(payload?.variants)) {
          setVariants(payload.variants);
        }
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="shadowPanel" aria-label="Shadow lab">
      <div className="shadowHeader">
        <div>
          <p className="eyebrow">SHADOW LAB · AUCUN PARI RÉEL</p>
          <h2>Variantes testées en parallèle</h2>
          <p>
            Elles observent les mêmes matchs sans apparaître comme paris.
            Aucune variante ne remplace le champion automatiquement.
          </p>
        </div>
      </div>

      {variants.length ? (
        <div className="shadowGrid">
          {variants.map((row) => (
            <article key={row.variant}>
              <div className="shadowTitle">
                <strong>{labels[row.variant] ?? row.variant}</strong>
                <span>{row.reviewEligible ? "À ÉTUDIER" : "COLLECTE"}</span>
              </div>
              <div className="shadowMetrics">
                <span>Réglés <b>{row.settled}</b></span>
                <span>ROI <b>{pct(row.roi)}</b></span>
                <span>CLV <b>{pct(row.avgClv)}</b></span>
                <span>DD <b>{row.maxDrawdownUnits.toFixed(1)}u</b></span>
              </div>
              <small>
                Revue possible uniquement après ≥100 paris réglés, ROI positif
                et ≥30 mesures CLV positives.
              </small>
            </article>
          ))}
        </div>
      ) : (
        <p className="shadowEmpty">
          Collecte démarrée : les premières variantes seront visibles dès que
          des matchs FULL ML satisfont leurs règles.
        </p>
      )}
    </section>
  );
}
